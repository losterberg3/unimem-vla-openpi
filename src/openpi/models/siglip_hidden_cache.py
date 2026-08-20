"""SigLIP with a causal hidden-state cache for incremental video frame processing.

Copy of siglip.py — original is untouched.

Note on naming: what's cached per temporal block is the raw pre-LN, pre-temporal-PE
*hidden state* (residual-stream activation) for past frames, not projected attention
key/value tensors — this block still computes its own Q/K/V fresh from the cache
every call. It is not a "KV cache" in the usual transformer-attention sense (that's
what the Gemma LLM's kv_cache in pi0.py's sample_actions is); it's a rolling buffer
of hidden states.

Key additions:
    TemporalStrideBlockCached  — like TemporalStrideBlock but processes one new
        frame per call, attending temporally against cached raw features from
        previous frames.  Sub-module declaration order matches Encoder1DBlock /
        TemporalStrideBlock so pretrained weights transfer by name.

    VideoEncoderCached  — like VideoEncoder but accepts a list of per-temporal-
        block hidden-state caches (plus a validity counter) and returns updated
        caches alongside the output for the new frame.

Cache format
------------
A ``cache`` is a list of ``jnp.ndarray`` with one entry per temporal block
(i.e. len(cache) == depth // temporal_stride). Each entry has a FIXED shape
(B*N, num_frames-1, D) — always this shape, every call, forever. There is no
"None" or growing-length state: slot 0 is always the oldest position, slot
num_frames-2 is always the most recent historical position, relative to
whichever frame is currently being processed as "new".

Because the shape never changes, seeding the cache from scratch (feeding
num_frames-1 placeholder frames, e.g. all-zero images, one at a time) and
every subsequent real inference call all hit the exact same JIT trace — no
per-call or per-seed-step recompilation.

Validity counter
-----------------
``valid_len`` (scalar int32) tracks how many of the num_frames-1 slots hold
real (already-computed) content, counted from the most-recent end. It starts
at 0 (freshly allocated cache — all slots are placeholder garbage) and
saturates at num_frames-1 after that many real calls. Slot j is treated as
real iff ``j >= (num_frames - 1) - valid_len``; the attention mask over the
cache is built from this condition, so garbage placeholder slots are never
actually attended to before they've been overwritten by real content — the
buffer shift itself is unconditional (it always evicts slot 0 and appends
the newest frame at the end) whether or not slot 0 was garbage.

PE handling
-----------
The cache stores raw pre-LN, pre-temporal-PE features. At attention time, the
correct sinusoidal PE for each slot is applied fresh, matching training's
[pad_0, pad_1, ..., pad_{Tm1-1}, current] buffer-position layout exactly:

``x`` (the frame currently being processed) occupies buffer position
``valid_len`` in the eventual num_frames-length window — during seeding this
is the zero-frame's true pad position (0, 1, 2, ...); once fully seeded
(valid_len == num_frames-1) it's always the final/"current" position. This is
NOT always ``num_frames-1`` — that's only correct after seeding completes;
using it unconditionally silently mismatches training's per-pad-frame PE
during the seeding steps themselves; see git history for how this manifested
(a real, measurable ~2-3% output divergence against the batched reference
even though the *final* seeded-cache result looked plausible).

Cache slot j's true buffer position, derived from the unconditional shift
(the newest entry, slot Tm1-1, is always at position ``valid_len``), is
``valid_len - Tm1 + j`` — negative values only occur in not-yet-real slots,
which the validity mask (below) excludes from attention anyway.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.training.sharding as sharding

# ---------------------------------------------------------------------------
# Reuse unchanged building blocks from siglip.py
# ---------------------------------------------------------------------------
from openpi.models.siglip import (
    MlpBlock,
    Encoder1DBlock,
    _sinusoidal_temporal_pe,
)


# ---------------------------------------------------------------------------
# Hidden-state-cache temporal block
# ---------------------------------------------------------------------------

class TemporalStrideBlockCached(nn.Module):
    """TemporalStrideBlock that processes one new frame against a fixed-size,
    validity-masked hidden-state cache.

    Sub-modules are declared in the same order as Encoder1DBlock so Flax
    auto-assigns matching names (LayerNorm_0, MultiHeadDotProductAttention_0,
    LayerNorm_1, MlpBlock_0) and pretrained SigLIP weights transfer directly.
    """

    num_frames: int = 4
    num_heads: int = 12
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, hidden_cache, valid_len, *, deterministic: bool = True):
        """
        Args:
            x:            (B, N, D)  tokens for the new frame — raw, no temporal PE.
            hidden_cache: (B*N, num_frames-1, D)  ALWAYS this fixed shape. Slot j is
                          age (num_frames-1-j) relative to `x`; may still hold
                          placeholder garbage in slots not yet covered by valid_len.
            valid_len:    scalar int32, how many of the num_frames-1 slots (counted
                          from the most-recent end) hold real content.
        Returns:
            x_out:            (B, N, D)
            new_hidden_cache: (B*N, num_frames-1, D)  same fixed shape.
        """
        B, N, D = x.shape
        Tm1 = self.num_frames - 1

        # Shared sub-modules — same declaration order as Encoder1DBlock / TemporalStrideBlock.
        attn_ln = nn.LayerNorm(dtype=self.dtype_mm)                    # LayerNorm_0
        attn = nn.MultiHeadDotProductAttention(                         # MultiHeadDotProductAttention_0
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )
        mlp_ln = nn.LayerNorm(dtype=self.dtype_mm)                     # LayerNorm_1
        mlp = MlpBlock(mlp_dim=self.mlp_dim, dropout=self.dropout, dtype_mm=self.dtype_mm)  # MlpBlock_0

        pe = _sinusoidal_temporal_pe(self.num_frames, D, dtype=x.dtype)  # (T, D)

        # --- Unconditional shift-and-append: always fixed shape (B*N, Tm1, D). ---
        raw_x = x.reshape(B * N, 1, D)
        new_hidden_cache = jnp.concatenate([hidden_cache[:, 1:, :], raw_x], axis=1)

        # --- Temporal attention ---
        # `x` occupies buffer position `valid_len` in the eventual num_frames-length
        # window: during seeding this is the zero-frame's true pad position
        # (0, 1, 2, ...); once fully seeded (valid_len == num_frames-1) it's always
        # the final/"current" position, matching training's [pad...,current] layout
        # exactly. Do NOT hardcode num_frames-1 here — that's only correct post-seed.
        x_pe = x + pe[valid_len][None, None, :]                        # (B, N, D)
        x_pe = sharding.activation_sharding_constraint(x_pe)
        x_flat = x_pe.reshape(B * N, 1, D)
        q_ln = attn_ln(x_flat)                                         # (B*N, 1, D)

        # Slot j holds a frame whose true buffer position is (valid_len - Tm1 + j)
        # relative to the eventual window (derived from the unconditional shift: the
        # newest entry, slot Tm1-1, is always at position `valid_len`). Clamped since
        # out-of-range values only occur in not-yet-real slots, which are masked below.
        pos_hist = jnp.clip(valid_len - Tm1 + jnp.arange(Tm1), 0, self.num_frames - 1)
        pe_hist = pe[pos_hist]                                         # (Tm1, D)
        kv_with_pe = hidden_cache + pe_hist[None, :, :]                # (B*N, Tm1, D)
        kv_ln = attn_ln(kv_with_pe)
        y_kv = jnp.concatenate([kv_ln, q_ln], axis=1)                  # (B*N, Tm1+1, D)

        # Mask out still-garbage (not-yet-seeded) history slots: slot j is real
        # iff j >= Tm1 - valid_len. The current-frame key (last position) is
        # always real.
        slot_idx = jnp.arange(Tm1)
        hist_valid = slot_idx >= (Tm1 - valid_len)                     # (Tm1,) bool
        full_valid = jnp.concatenate([hist_valid, jnp.ones((1,), dtype=jnp.bool_)])
        mask = full_valid[None, None, None, :]                        # (1, 1, 1, Tm1+1)

        y = attn(q_ln, y_kv, mask=mask)                                # (B*N, 1, D)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = (x_flat + y).reshape(B, N, D)                             # residual → (B, N, D)

        # --- Spatial attention (shared QKV weights, single frame) ---
        y = attn_ln(x)
        y = attn(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = sharding.activation_sharding_constraint(x + y)

        # --- MLP ---
        y = mlp_ln(x)
        y = mlp(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = sharding.activation_sharding_constraint(x + y)

        return x, new_hidden_cache


# ---------------------------------------------------------------------------
# Full encoder with cache management
# ---------------------------------------------------------------------------

class VideoEncoderCached(nn.Module):
    """VideoEncoder variant for incremental single-frame inference with a
    fixed-size, validity-masked hidden-state cache.

    Layer names match VideoEncoder (``encoderblock_0 … encoderblock_{depth-1}``)
    so weights can be shared / transferred.
    """

    depth: int = 27
    num_frames: int = 4
    temporal_stride: int = 4
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, caches: list, valid_len, *, deterministic: bool = True):
        """
        Args:
            x:         (B, N, D)  tokens for the new frame (after patch embed + spatial posemb).
            caches:    list of (B*N, num_frames-1, D) fixed-shape per-temporal-block
                       caches. Length must equal depth // temporal_stride.
            valid_len: scalar int32, shared across all temporal blocks (they all
                       advance in lockstep, one real frame per call).
        Returns:
            encoded:       (B, N, D)  output for the new frame.
            new_caches:    list of updated caches, same fixed shapes as `caches`.
            new_valid_len: scalar, min(valid_len + 1, num_frames - 1).
        """
        remat_kwargs = {
            "prevent_cse": False,
            "static_argnums": (2,),  # deterministic
            "policy": getattr(jax.checkpoint_policies, self.remat_policy, None),
        }
        RematSpatial  = nn.remat(Encoder1DBlock,            **remat_kwargs)
        RematTemporal = nn.remat(TemporalStrideBlockCached, **remat_kwargs)

        new_caches = []
        cache_idx = 0
        for lyr in range(self.depth):
            if (lyr + 1) % self.temporal_stride == 0:
                block = RematTemporal(
                    name=f"encoderblock_{lyr}",
                    num_frames=self.num_frames,
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, new_cache = block(x, caches[cache_idx], valid_len, deterministic=deterministic)
                new_caches.append(new_cache)
                cache_idx += 1
            else:
                block = RematSpatial(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, _ = block(x, deterministic)

        new_valid_len = jnp.minimum(valid_len + 1, self.num_frames - 1)
        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), new_caches, new_valid_len
