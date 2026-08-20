# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


# Addition (not in the original PI repo): reshape+matmul patch embedding for GPU compat.
class _PatchEmbedMatmul(nn.Module):
    """Patch embedding via reshape+matmul instead of Conv2D.

    Uses the same parameter names/shapes as nn.Conv so checkpoints are compatible.
    Avoids cudnnConvolutionBackwardFilter, which fails on some GPU architectures.
    """

    width: int
    patch_size: Sequence[int]
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        n, h_img, w_img, c = x.shape
        ph, pw = self.patch_size
        h, w = h_img // ph, w_img // pw

        # Same param names/shapes as nn.Conv → checkpoint-compatible
        kernel = self.param("kernel", nn.initializers.lecun_normal(), (ph, pw, c, self.width), self.dtype)
        bias = self.param("bias", nn.initializers.zeros, (self.width,), self.dtype)

        # Non-overlapping patch extraction via reshape (no cuDNN involved)
        patches = x.reshape(n, h, ph, w, pw, c).transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(n, h, w, ph * pw * c)

        return jnp.einsum("nhwp,pd->nhwd", patches, kernel.reshape(-1, self.width)) + bias


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y

        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # Input Encoder
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


# ---------------------------------------------------------------------------
# Additions (not in the original PI repo): temporal attention for video input.
# `TemporalStrideBlock` reuses `Encoder1DBlock`'s spatial QKV weights for a
# causal cross-frame attention pass, interleaved every `temporal_stride`
# layers by `VideoEncoder`. See siglip_hidden_cache.py for the streaming
# (one-frame-at-a-time, cached) counterpart used at inference time.
# ---------------------------------------------------------------------------


def _sinusoidal_temporal_pe(T: int, D: int, dtype=jnp.float32) -> jnp.ndarray:
    """Fixed sinusoidal PE matching MEM Appendix C: e(0)=0 for current frame.

    Pure-sine so position 0 → zero vector. Buffer index 0=oldest gets position
    T-1; buffer index T-1=current frame gets position 0. Returns (T, D).
    """
    positions = jnp.arange(T - 1, -1, -1, dtype=jnp.float32)  # T-1, ..., 1, 0
    dim_indices = jnp.arange(D, dtype=jnp.float32)
    angles = positions[:, None] / (10_000 ** (dim_indices[None, :] / D))
    return jnp.sin(angles).astype(dtype)


class TemporalStrideBlock(nn.Module):
    """Pi-MEM temporal stride block: temporal → spatial attention (shared QKV) → MLP.

    Sub-modules are declared in the same order as Encoder1DBlock so Flax
    auto-assigns matching names (LayerNorm_0, MultiHeadDotProductAttention_0,
    LayerNorm_1, MlpBlock_0).  This lets pretrained SigLIP weights transfer
    directly — Flax resolves params by name, not class type.

    Both temporal and spatial attention reuse the same LN and QKV weights
    """

    num_frames: int = 4
    num_heads: int = 12
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        BT, N, D = x.shape
        T = self.num_frames
        B = BT // T

        # Add temporal PE: e(0)=0 for current frame, e(T-1) for oldest.
        pe = _sinusoidal_temporal_pe(T, D, dtype=x.dtype)  # (T, D)
        x = (x.reshape(B, T, N, D) + pe[None, :, None, :]).reshape(BT, N, D)
        x = sharding.activation_sharding_constraint(x)

        # Declare shared sub-modules in the same order as Encoder1DBlock so
        # Flax auto-assigns matching names for checkpoint compatibility.
        attn_ln = nn.LayerNorm(dtype=self.dtype_mm)           # LayerNorm_0
        attn = nn.MultiHeadDotProductAttention(                # MultiHeadDotProductAttention_0
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )
        mlp_ln = nn.LayerNorm(dtype=self.dtype_mm)            # LayerNorm_1
        mlp = MlpBlock(                                        # MlpBlock_0
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )

        # 1. Temporal attention (causal): reshape to (B*N, T, D), attend across frames.
        x_t = x.reshape(B, T, N, D).transpose(0, 2, 1, 3).reshape(B * N, T, D)
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, None, :, :]
        y = attn_ln(x_t)
        y = attn(y, y, mask=causal_mask)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x_t = x_t + y
        x = x_t.reshape(B, N, T, D).transpose(0, 2, 1, 3).reshape(BT, N, D)

        # 2. Spatial attention (same QKV weights): attend within each frame.
        y = attn_ln(x)
        y = attn(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = sharding.activation_sharding_constraint(x + y)

        # 3. MLP.
        y = mlp_ln(x)
        y = mlp(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = sharding.activation_sharding_constraint(x + y)

        return x, {}


class VideoEncoder(nn.Module):
    """Spatial encoder with interleaved causal temporal attention.

    Every ``temporal_stride`` spatial layers, the same spatial block is reused
    for a causal temporal pass (reshaping tokens to attend across frames). No new
    parameters are introduced — temporal attention shares weights with the spatial
    blocks, exactly as in Pi-MEM.

    NOTE: the pretrained SigLIP checkpoint stores spatial weights in scan format
    (stacked under a single ``encoderblock`` key).  Loading into this module
    requires a one-time conversion that unstacks those arrays into
    ``encoderblock_0 … encoderblock_{depth-1}``.
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
    def __call__(self, x, *, deterministic=True):
        # x: (B*T, N, D) where T == self.num_frames.
        remat_kwargs = {
            "prevent_cse": False,
            "static_argnums": (2,),
            "policy": getattr(jax.checkpoint_policies, self.remat_policy, None),
        }
        RematSpatial  = nn.remat(Encoder1DBlock,      **remat_kwargs)
        RematTemporal = nn.remat(TemporalStrideBlock, **remat_kwargs)

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
            else:
                block = RematSpatial(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
            x, _ = block(x, deterministic)

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), {}


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    # Video encoder: if > 1, image is (B, T, H, W, C) and temporal attention is
    # interleaved every temporal_stride spatial layers.
    num_frames: int = 1
    temporal_stride: int = 4

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # A temporal axis means video_encoder=True upstream (Pi0Config always
        # builds a 5D [B,T,H,W,C] fake_obs in that case, even for num_frames=1)
        # — not self.num_frames > 1, which wrongly conflates "T==1" with
        # "no temporal axis at all" (true only for video_encoder=False, where
        # the image is 4D from the start). Fold T into the batch dim.
        is_video = image.ndim == 5
        if is_video:
            B, T, H, W, C = image.shape
            image = jnp.reshape(image, (B * T, H, W, C))

        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        image = jnp.asarray(image, jnp.float32)

        # Patch extraction — uses reshape+matmul to avoid cudnnConvolutionBackwardFilter
        # (which fails on Blackwell GPUs); parameter names match nn.Conv for ckpt compat.
        x = out["stem"] = _PatchEmbedMatmul(
            self.width,
            self.patch_size,
            dtype=jnp.float32,
            name="embedding",
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        if is_video:
            x, out["encoder"] = VideoEncoder(
                depth=self.depth,
                num_frames=self.num_frames,
                temporal_stride=self.temporal_stride,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                remat_policy=self.remat_policy,
                dtype_mm=self.dtype_mm,
                name="Transformer",
            )(x, deterministic=not train)
        else:
            x, out["encoder"] = Encoder(
                depth=self.depth,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                scan=self.scan,
                remat_policy=self.remat_policy,
                dtype_mm=self.dtype_mm,
                name="Transformer",
            )(x, deterministic=not train)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
