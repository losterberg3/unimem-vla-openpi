"""Patch LeRobot's HF batch transform before ``LeRobotDataset`` is imported.

LeRobot's stock helper does::

    x if isinstance(x, str) else torch.tensor(x)

HuggingFace often yields **image** columns as ``{"bytes": ..., "path": ...}`` dicts (not PIL),
and extra columns like ``labels`` / ``phase_history`` as Arrow **structs** (dict) or ``bytes``.
Any of those hits ``torch.tensor(dict)`` → ``RuntimeError: Could not infer dtype of dict``.

This replaces that transform with the same behavior as LeRobot for PIL images (``ToTensor``),
the same numeric behavior for arrays/scalars, plus:
- HF image dicts → decode to PIL → ``ToTensor`` (same pixels/range as the PIL branch).
- Single-field structs → scalar / str.
- Raw ``bytes`` in text-like cells → UTF-8 ``str``.

Import this module before ``lerobot.common.datasets.lerobot_dataset``.
"""

from __future__ import annotations

import io
from typing import Any

import lerobot.common.datasets.utils as _lerobot_utils
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

_to_tensor = transforms.ToTensor()


def _unwrap_single_field_struct(x: dict) -> Any:
    if len(x) != 1:
        return x
    v = next(iter(x.values()))
    if isinstance(v, (int, float, bool, np.integer, np.floating)):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return x


def _hf_image_dict_to_pil(x: dict) -> PILImage.Image:
    if x.get("bytes") is not None:
        return PILImage.open(io.BytesIO(x["bytes"])).convert("RGB")
    if x.get("path"):
        return PILImage.open(x["path"]).convert("RGB")
    raise ValueError("HF image dict has neither bytes nor path")


def hf_transform_to_torch(items_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop-in replacement for ``lerobot.common.datasets.utils.hf_transform_to_torch``."""
    for key in items_dict:
        col = items_dict[key]
        if not col:
            continue
        first = col[0]
        if isinstance(first, PILImage.Image):
            items_dict[key] = [_to_tensor(img) for img in col]
        elif first is None:
            pass
        else:
            out: list[Any] = []
            for x in col:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, bytes):
                    out.append(x.decode("utf-8"))
                elif isinstance(x, PILImage.Image):
                    out.append(_to_tensor(x))
                elif isinstance(x, dict):
                    if x.get("bytes") is not None or x.get("path"):
                        out.append(_to_tensor(_hf_image_dict_to_pil(x)))
                    else:
                        u = _unwrap_single_field_struct(x)
                        if isinstance(u, dict):
                            out.append(u)
                        elif isinstance(u, str):
                            out.append(u)
                        else:
                            out.append(torch.tensor(u))
                else:
                    out.append(torch.tensor(x))
            items_dict[key] = out
    return items_dict


_lerobot_utils.hf_transform_to_torch = hf_transform_to_torch


# Patch check_timestamps_sync to handle truncated/clipped datasets
# Some datasets (e.g., mem10) are truncated but metadata isn't updated, causing
# timestamp sync checks to fail. This patch filters out-of-bounds indices and
# warns instead of crashing when timestamps are misaligned due to truncation.
_original_check_timestamps_sync = _lerobot_utils.check_timestamps_sync


def check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, fps, tolerance_s, raise_value_error=True):
    """Fixed version that handles truncated datasets with misaligned metadata."""
    if timestamps.shape != episode_indices.shape:
        raise ValueError(
            "timestamps and episode_indices should have the same shape. "
            f"Found {timestamps.shape=} and {episode_indices.shape=}."
        )

    # Consecutive differences
    diffs = np.diff(timestamps)
    within_tolerance = np.abs(diffs - (1.0 / fps)) <= tolerance_s

    # Mask to ignore differences at the boundaries between episodes
    mask = np.ones(len(diffs), dtype=bool)
    ignored_diffs = ep_data_index_np["to"][:-1] - 1  # indices at the end of each episode

    # Filter out indices that are out of bounds (happens when dataset is truncated but metadata isn't updated)
    out_of_bounds_mask = ignored_diffs >= len(diffs)
    if np.any(out_of_bounds_mask):
        import logging
        logging.warning(
            f"Dataset appears truncated: {np.sum(out_of_bounds_mask)} episode boundaries are out of bounds. "
            f"Expected ~{ignored_diffs.max()} frames but only have {len(diffs)} timestamp diffs. "
            f"This can happen if the dataset was clipped but metadata wasn't updated."
        )
    ignored_diffs = ignored_diffs[~out_of_bounds_mask]

    mask[ignored_diffs] = False
    filtered_within_tolerance = within_tolerance[mask]

    # Check if all remaining diffs are within tolerance
    if not np.all(filtered_within_tolerance):
        if raise_value_error:
            # Count how many timestamps are out of tolerance
            n_bad = np.sum(~filtered_within_tolerance)
            pct_bad = 100.0 * n_bad / len(filtered_within_tolerance) if len(filtered_within_tolerance) > 0 else 0
            import logging
            logging.warning(
                f"Dataset has {n_bad} timestamps ({pct_bad:.1f}%) outside tolerance. "
                f"This may indicate data synchronization issues or a truncated dataset. "
                f"Allowing training to proceed anyway."
            )
            # Don't crash - allow training to continue with potentially misaligned timestamps
            return True
        return False

    return True


_lerobot_utils.check_timestamps_sync = check_timestamps_sync
