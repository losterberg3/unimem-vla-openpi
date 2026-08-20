import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # If the reference params use named sequential SigLIP blocks (video encoder),
        # the base checkpoint has them stacked under a single "encoderblock" key (scan
        # format).  Convert before the flat-key intersection in _merge_params so the
        # spatial weights actually land in the right places.
        flat_ref = flax.traverse_util.flatten_dict(params)
        needs_sequential = any("encoderblock_0" in str(k) for k in flat_ref)
        if needs_sequential:
            loaded_params = _model.convert_siglip_scan_to_sequential(loaded_params)

        # Fill gaps from the reference tree: LoRA/phase_head (not in released checkpoints)
        # and temporal attention blocks (new for video encoder) fall back to the freshly
        # initialized reference model's own init values — there's no dedicated init-time
        # gate anymore (removed from TemporalStrideBlock/TemporalStrideBlockCached).
        return _merge_params(loaded_params, params, missing_regex=r".*(lora|phase_head|temporal).*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params)
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params)

    def _cast_to_ref_dtype_if_possible(v, ref_v):
        if not hasattr(v, "dtype") or not hasattr(ref_v, "dtype"):
            return v
        if v.dtype == ref_v.dtype:
            return v
        # Some leaves can carry non-standard dtypes (e.g. PRNG key dtype ``key<fry>``)
        # that numpy cannot interpret/cast via ``astype``.
        try:
            np.dtype(ref_v.dtype)
        except TypeError:
            return v
        try:
            return v.astype(ref_v.dtype)
        except (TypeError, ValueError):
            return v

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        # Keep runtime RNG state from the freshly initialized reference tree.
        if any(str(part) == "rngs" for part in k):
            continue
        if k in flat_ref:
            result[k] = _cast_to_ref_dtype_if_possible(v, flat_ref[k])

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    def _keypath_to_str(keypath: tuple[object, ...]) -> str:
        return "/".join(str(part) for part in keypath)

    for k in {k for k in flat_ref if pattern.fullmatch(_keypath_to_str(k))}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result)
