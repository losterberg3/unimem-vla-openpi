import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


def _align_params_to_expected_shape(expected: at.Params, loaded: at.Params) -> at.Params:
    """Use checkpoint weights when shapes match; otherwise keep initialized leaves as jax arrays."""
    # Flatten without ``sep``: NNX modules (e.g. ``nnx.Sequential``) use int path segments; ``sep="/"`` would fail.
    flat_e = traverse_util.flatten_dict(expected)
    flat_l = traverse_util.flatten_dict(loaded)
    merged: dict = {}

    def _jax_leaf_for_mismatch(e) -> jax.Array:
        """Concrete fallback leaf as ``jax.Array`` for consistent runtime behavior."""
        cpu = jax.devices("cpu")[0]
        if isinstance(e, jax.Array):
            if jnp.issubdtype(e.dtype, jax.dtypes.prng_key):
                return jax.device_put(jnp.zeros(e.shape, dtype=e.dtype), cpu)
            return jnp.zeros(e.shape, dtype=e.dtype)
        if isinstance(e, np.ndarray):
            return jnp.asarray(e)
        shape = getattr(e, "shape", None)
        dtype = getattr(e, "dtype", None)
        if shape is None or dtype is None:
            raise TypeError(f"Cannot build host fallback leaf from {type(e)}: {e!r}")
        shape_t = tuple(int(s) for s in shape)
        if jnp.issubdtype(dtype, jax.dtypes.prng_key):
            return jax.device_put(jnp.zeros(shape_t, dtype=dtype), cpu)
        return jnp.zeros(shape_t, dtype=dtype)

    for k, e in flat_e.items():
        if k not in flat_l:
            # Missing in checkpoint: materialize eval-shape leaves to concrete arrays.
            merged[k] = _jax_leaf_for_mismatch(e)
            continue
        l = flat_l[k]
        l_shape = tuple(getattr(l, "shape", ()))
        l_dtype = getattr(l, "dtype", None)
        e_shape = getattr(e, "shape", None)
        e_dtype = getattr(e, "dtype", None)
        if e_shape is not None and tuple(e_shape) == l_shape:
            # Force all matched leaves to JAX arrays for fast execution.
            merged[k] = l if isinstance(l, jax.Array) else jnp.asarray(l)
        else:
            logger.warning(
                "Checkpoint param %s: expected shape %s %s, got %s %s; using initialized value",
                "/".join(str(p) for p in k) if isinstance(k, tuple) else str(k),
                e_shape,
                e_dtype,
                l_shape,
                l_dtype,
            )
            merged[k] = _jax_leaf_for_mismatch(e)
    return traverse_util.unflatten_dict(merged)


def convert_siglip_scan_to_sequential(params: "at.Params", depth: int = 27) -> "at.Params":
    """Convert scan-stacked SigLIP encoder params to per-layer sequential format.

    Pretrained SigLIP checkpoints (scan=True) store all encoder-block parameters
    under a single ``"encoderblock"`` key whose leaves have shape ``(depth, ...)``.
    ``VideoEncoder`` expects individual ``"encoderblock_0"`` … ``"encoderblock_{depth-1}"``
    keys with leaves of shape ``(...,)``.  This function does that one-time conversion so
    the pretrained spatial weights load correctly.

    Temporal attention block params are new and may be absent from an older checkpoint;
    ``_align_params_to_expected_shape`` fills any such missing leaf with a zero-initialized
    fallback (there's no dedicated init-time gate anymore — that was removed from
    ``TemporalStrideBlock``/``TemporalStrideBlockCached``).
    """

    def _recurse(d: dict) -> dict:
        result = {}
        for k, v in d.items():
            if k == "encoderblock" and isinstance(v, dict):
                for lyr in range(depth):
                    result[f"encoderblock_{lyr}"] = jax.tree.map(lambda x, i=lyr: x[i], v)
            elif isinstance(v, dict):
                result[k] = _recurse(v)
            else:
                result[k] = v
        return result

    return _recurse(params)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
#
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#   v = event id vocabulary size
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    # Uses *b_img (not *b) so that video inputs (B, T, H, W, C) don't conflict
    # with state/prompt which always have batch dim B only.
    images: dict[str, at.Float[ArrayT, "*b_img h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b_img"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Pre-encoded image tokens (B, L, D) projected to LLM width — bypasses SigLIP when set.
    # Used by the hidden-state-cached inference path in Policy so the full VideoEncoder is
    # skipped for all but the first call; VideoEncoderCached runs externally and injects tokens here.
    pre_encoded_images: dict | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]
# format of text tokens produced by inference
Text = at.Int[ArrayT, "*b t"]
# event tracking models
Labels = at.Int[ArrayT, "*b v"]

def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] | None = None,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if image_keys is None:
        image_keys = list(observation.images.keys())

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # Video: (B, T, H, W, C) → fold T into batch so resize/augment work on individual frames
        is_video = image.ndim == 5
        if is_video:
            B_vid, T_vid = image.shape[:2]
            image = jnp.reshape(image, (B_vid * T_vid, *image.shape[2:]))

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        if is_video:
            image = jnp.reshape(image, (B_vid, T_vid, *image.shape[1:]))

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        pre_encoded_images=observation.pre_encoded_images,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        params = _align_params_to_expected_shape(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: ``np.ndarray`` restores to host RAM (often safer for large single-GPU loads); ``jax.Array`` uses
            replicated sharding across ``jax.devices()`` by default.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: Optional sharding when ``restore_type`` is ``jax.Array``.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
