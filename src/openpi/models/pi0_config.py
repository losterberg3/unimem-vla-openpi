import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    """Config for `Pi0` (see that class's docstring for the model itself).

    `event_tracking` and `video_encoder` are this fork's two additions on top of
    upstream Pi0/Pi0.5, and are independently toggleable — see `video_encoder`'s
    docstring below for exactly how (and how not) they combine.
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    event_tracking: bool = False
    action_samples: int = 1
    # Video encoder: set True to ingest T frames per camera instead of 1, via SigLIP
    # temporal attention (see models/siglip.py's TemporalStrideBlock/VideoEncoder).
    # Set num_frames to the desired temporal depth (e.g. 4); default of 1 matches the
    # single-image baseline.
    #
    # This flag is purely about the MODEL: it says nothing about how those T frames
    # were selected — that's entirely a training/config.py DataConfigFactory choice,
    # and the two are not symmetric:
    #   - Fixed time stride ("naive video", LeRobotLiberoDataConfig /
    #     LeRobotXarmVideoDataConfig): no event tracking, ever. Video never combines
    #     with event_tracking=True.
    #   - Actual past event-transition frames ("keyframes", *EventKeyframeDataConfig,
    #     via EventMemoryDataset): always paired with event_tracking=True — this is
    #     the only way video_encoder=True and event_tracking=True combine.
    # Both train the exact same video_encoder=True architecture; only the data
    # pipeline (and whether event_tracking is even on) differs. See the README's
    # "Training your own event-memory policy" section.
    video_encoder: bool = False
    num_frames: int = 1
    # Override for the SigLIP encoder's `scan` flag (jax.lax.scan over the 27
    # encoder blocks vs a plain unrolled loop). None (default) preserves
    # existing behavior: scan=(not video_encoder). Scanning trades some
    # runtime speed for lower compile time/binary size; set this explicitly
    # to isolate that tradeoff from video_encoder when benchmarking latency
    # (video_encoder=False models otherwise can't use the unrolled path).
    scan: bool | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        if self.video_encoder:
            image_spec = jax.ShapeDtypeStruct(
                [batch_size, self.num_frames, *_model.IMAGE_RESOLUTION, 3], jnp.float32
            )
        else:
            image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        if self.video_encoder:
            image_mask_spec = jax.ShapeDtypeStruct([batch_size, self.num_frames], jnp.bool_)
        else:
            image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True):
        """Load checkpoint params, auto-converting scan→sequential format for the video encoder."""
        if self.video_encoder:
            params = _model.convert_siglip_scan_to_sequential(params, depth=27)
        return super().load(params, remove_extra_params=remove_extra_params)

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
