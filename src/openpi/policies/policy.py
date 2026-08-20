from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.linen as nn
import flax.traverse_util
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.siglip import _PatchEmbedMatmul
from openpi.models.siglip_hidden_cache import VideoEncoderCached
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

logger = logging.getLogger("openpi")


class Policy(BasePolicy):
    """Wraps a model + input/output transforms behind the `BasePolicy.infer()` interface
    used by `WebsocketPolicyServer` and the simple in-process clients.

    Beyond the base fine-tuning path, this fork's `Policy` also:
      - Returns an `event_id` field (predicted event-class probabilities) in
        `infer()`'s output whenever the model has `event_tracking=True`.
      - For `video_encoder=True` models trained on event-triggered keyframes (not
        fixed-stride "naive video" — see `Pi0Config.video_encoder`'s docstring),
        `infer()` can run the SigLIP hidden-state cache (`siglip_hidden_cache.py`)
        internally, so callers only ever send the CURRENT frame per camera plus a
        `new_keyframe` flag on the calls that should slide it. Call `reset_cache()`
        once at the start of each rollout, or the cache carries stale history across
        episodes. Naive-video-trained models are served differently: the CLIENT
        stacks the full frame history itself and sends it every call (no
        `reset_cache`/hidden-state cache involved) — mixing the two up serves frames
        in an order the model wasn't trained on.
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions
                (or model.sample_actions_event, for event-tracking models).
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        # SigLIP hidden-state cache state (None = disabled; dict = active, per-camera
        # (caches, valid_len) pairs — see siglip_hidden_cache.py for the fixed-size,
        # validity-masked cache format).
        self._siglip_caches: dict[str, tuple[list, jax.Array]] | None = None
        self._siglip_encode_fn = None   # JIT-compiled encode fn, built lazily
        self._n_temporal_blocks: int = 0

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup: bind whichever sample fn `infer()` will call below —
            # `sample_actions_event` also returns event logits, `sample_actions` doesn't.
            if model.event_tracking:
                self._sample_actions_event = nnx_utils.module_jit(model.sample_actions_event)
            else:
                self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng if rng is not None else jax.random.key(0)

    def reset_cache(self) -> None:
        """Reset the SigLIP hidden-state cache. Call at the start of each video-mode rollout."""
        if not (hasattr(self._model, "video_encoder") and self._model.video_encoder):
            return
        if self._siglip_encode_fn is None:
            self._build_siglip_encode_fn()
            self._warmup_siglip_fns()
        # Empty dict; per-camera caches are lazily created in infer() on first
        # appearance (Pi0 doesn't expose fake_obs() — that's a config-level method).
        self._siglip_caches = {}
        logger.info("SigLIP hidden-state cache reset (per-camera caches will initialize on first frame)")

    def _warmup_siglip_fns(self) -> None:
        """Force the one JIT compile right here, once, so no real call — including
        the seeding calls and step 0 — ever pays a first-call compile tax. Runs
        exactly once per server lifetime, guarded by the same `_siglip_encode_fn
        is None` check as the build step. The cache is always this same fixed
        shape (see siglip_hidden_cache.py), so this single trace covers every future
        call: seeding, sliding, and re-encoding alike."""
        dummy = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
        dummy_cache = self._empty_siglip_cache()
        self._siglip_encode_fn(dummy, dummy_cache, jnp.int32(0))

    def _empty_siglip_cache(self) -> list:
        """A freshly-allocated, fixed-shape placeholder cache (all garbage,
        valid_len=0) — content doesn't matter since valid_len masks it out.
        Shape is (B*N_patches, num_frames-1, width); infer() always batches to
        B=1 before this point, so B*N_patches == N_patches here."""
        Tm1 = self._num_frames - 1
        return [
            jnp.zeros((self._n_patches, Tm1, self._width), dtype=jnp.bfloat16)
            for _ in range(self._n_temporal_blocks)
        ]

    def _build_siglip_encode_fn(self) -> None:
        """Build a JIT-compiled VideoEncoderCached encode function from model weights.

        Extracts SigLIP params from the NNX-wrapped Linen module, infers model
        dimensions from param shapes, then compiles a function that:
          (1) patch-embeds + spatial-posemb a single (B, H, W, C) frame,
          (2) runs VideoEncoderCached with the rolling hidden-state cache,
          (3) projects with the head Dense to LLM width.
        """
        model = self._model

        # Extract Linen params from the NNX-wrapped SigLIP module.
        _, img_state = nnx.split(model.PaliGemma.img)
        siglip_p = img_state.to_pure_dict()
        # Strip the outer "params" collection key if present.
        if "params" in siglip_p:
            siglip_p = siglip_p["params"]
        # siglip_p: {"embedding": ..., "pos_embedding": ..., "Transformer": ..., "head": ...}

        # Infer architecture dimensions from param shapes rather than hardcoding.
        pos_emb_arr = siglip_p["pos_embedding"]              # (1, N_patches, width)
        N_patches = pos_emb_arr.shape[1]                      # 256 for So400m/14
        width = pos_emb_arr.shape[2]                          # 1152
        num_classes = siglip_p["head"]["kernel"].shape[1]     # LLM width, e.g. 2048
        n_blocks = sum(1 for k in siglip_p["Transformer"] if k.startswith("encoderblock_"))
        temporal_stride = 4  # fixed in our training setup
        n_temporal_blocks = n_blocks // temporal_stride
        patch_size = round((224 * 224 / N_patches) ** 0.5)  # 14 for 256 patches
        num_frames = model.num_frames

        # Hold param references — same JAX arrays as in the model (no copies).
        _embed_p = {"params": siglip_p["embedding"]}
        _enc_p = {"params": siglip_p["Transformer"]}
        _head_p = {"params": siglip_p["head"]}
        _pos_emb = pos_emb_arr  # (1, N_patches, width)

        enc_mod = VideoEncoderCached(
            depth=n_blocks,
            num_frames=num_frames,
            temporal_stride=temporal_stride,
            mlp_dim=4304,   # So400m constant; could infer from MLP block shape if needed
            num_heads=16,   # So400m constant
            dtype_mm="bfloat16",
        )

        @jax.jit
        def _encode(frame, caches, valid_len):
            """Encode one (B, H, W, C) frame against the fixed-shape, validity-masked
            hidden-state cache. Returns (tokens, new_caches, new_valid_len). Cache shape never
            changes — seeding (zero frames, valid_len 0 -> num_frames-1) and every
            real call afterward (valid_len pinned at num_frames-1) hit this same trace."""
            x = _PatchEmbedMatmul(width, (patch_size, patch_size), dtype=jnp.float32, name="embedding").apply(
                _embed_p, frame
            )  # (B, h_p, w_p, width)
            B, h_p, w_p, _ = x.shape
            x = jnp.reshape(x, (B, h_p * w_p, width))          # (B, N_patches, width)
            x = (x + _pos_emb).astype(jnp.bfloat16)             # add spatial posemb, cast
            encoded, new_caches, new_valid_len = enc_mod.apply(_enc_p, x, caches, valid_len)
            # Project to LLM width via the same head Dense used in the full SigLIP forward pass.
            tokens = nn.Dense(num_classes, dtype=jnp.bfloat16, name="head").apply(_head_p, encoded)
            return tokens, new_caches, new_valid_len  # (B, N_patches, num_classes), list[cache], scalar

        self._siglip_encode_fn = _encode
        self._n_temporal_blocks = n_temporal_blocks
        self._num_frames = num_frames
        self._n_patches = N_patches
        self._width = width

        logger.debug(
            "SigLIP hidden-state cache encode fn built: depth=%d temporal_blocks=%d N_patches=%d width=%d→%d",
            n_blocks, n_temporal_blocks, N_patches, width, num_classes,
        )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # Handle cache-reset signal before the input transform strips unknown keys.
        if inputs.pop("reset_cache", False):
            self.reset_cache()
        # Event-gated modes (text_keyframe/keyframe/gated) only slide the cache on the
        # call right after the client detects a new event; otherwise the current
        # slot is re-encoded fresh against the *unchanged* cached history.
        should_slide = inputs.pop("new_keyframe", False)

        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        # SigLIP hidden-state cache: pre-encode only the current frame for each camera,
        # then inject into the Observation so embed_prefix skips the full VideoEncoder.
        if self._siglip_caches is not None and not self._is_pytorch_model:
            pre_encoded = {}
            for name in list(observation.images.keys()):
                img = observation.images[name]
                # img is (B, H, W, C) when client sends a single frame,
                # or (B, T, H, W, C) if the client still stacks frames — take the last.
                curr_frame = img[:, -1, :, :] if img.ndim == 5 else img
                curr_frame = jnp.asarray(curr_frame, dtype=jnp.float32)
                if name not in self._siglip_caches:
                    # Seed with (num_frames - 1) zero-image calls through the exact
                    # same encode function real inference uses — the cache is always
                    # the same fixed shape and valid_len is a runtime value, not a
                    # shape, so this never triggers a second JIT trace. Matches the
                    # zero-padded history training actually saw, not "no history".
                    caches = self._empty_siglip_cache()
                    valid_len = jnp.int32(0)
                    # Images are already normalized to [-1, 1] by Observation.from_dict
                    # (raw_pixel/255*2-1) by this point, and training's zero-padding
                    # (EventMemoryDataset) used raw all-zero (black) pixel frames — a
                    # zero *pixel* value normalizes to -1.0, not 0.0. Must match that,
                    # not "zero in already-normalized space" (which is mid-gray).
                    zero_frame = jnp.full_like(curr_frame, -1.0)
                    for _ in range(self._num_frames - 1):
                        _, caches, valid_len = self._siglip_encode_fn(zero_frame, caches, valid_len)
                    self._siglip_caches[name] = (caches, valid_len)
                caches, valid_len = self._siglip_caches[name]
                tokens, new_caches, new_valid_len = self._siglip_encode_fn(
                    curr_frame, caches, valid_len,
                )
                pre_encoded[name] = tokens
                # Only persist the slide (evict-oldest + append-current) when the client
                # signals a real keyframe event; otherwise keep history unchanged so the
                # cache shape — and therefore the JIT trace — stays stable call to call.
                if should_slide:
                    self._siglip_caches[name] = (new_caches, new_valid_len)

            observation = _model.Observation(
                images=observation.images,
                image_masks=observation.image_masks,
                state=observation.state,
                tokenized_prompt=observation.tokenized_prompt,
                tokenized_prompt_mask=observation.tokenized_prompt_mask,
                token_ar_mask=observation.token_ar_mask,
                token_loss_mask=observation.token_loss_mask,
                pre_encoded_images=pre_encoded,
            )

        start_time = time.monotonic()

        if self._model.event_tracking:
            actions, event_id = self._sample_actions_event(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        # JAX dispatches GPU work asynchronously: the calls above return as soon as
        # the computation is *enqueued*, not once it's finished. Without this,
        # model_time below would only measure dispatch overhead (a couple ms) —
        # the real compute time doesn't show up until something actually blocks on
        # the result, which used to happen later (in the np.asarray conversion),
        # after model_time had already been recorded.
        if not self._is_pytorch_model:
            jax.block_until_ready(actions)

        outputs = {
            "state": inputs["state"],
            "actions": actions[None, ...],
        }

        if self._model.event_tracking:
            outputs["event_id"] = event_id

        model_time = time.monotonic() - start_time

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        def _cast_msgpack_compatible(x):
            arr = np.asarray(x)
            # msgpack numpy extension here cannot serialize bfloat16; normalize floats to fp32.
            if arr.dtype == np.dtype("bfloat16") or np.issubdtype(arr.dtype, np.floating):
                return arr.astype(np.float32)
            return arr

        outputs = jax.tree.map(_cast_msgpack_compatible, outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
