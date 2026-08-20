import json
import logging
import os
import pathlib
from typing import Any

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.

    Raises:
        ValueError: if `checkpoint_dir/assets/<TRAIN_SHAPE_FILE>` exists and its recorded
            model shape (video_encoder, num_frames, event_tracking, ...) doesn't match
            `train_config.model` — see `_checkpoints._shape_signature`. This guards against
            silently serving a checkpoint under settings it wasn't trained with (e.g. wrong
            `num_frames`, which changes the SigLIP hidden-state cache depth).
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Guard against serving a checkpoint under settings it wasn't trained with, BEFORE
    # loading any weights. num_frames in particular sets the hidden-state cache depth and
    # the temporal PE table — a mismatch loads and runs silently while showing the model a
    # different history layout than it was fit to. Checkpoints predating this record are
    # skipped, so this is backwards compatible.
    _shape_file = pathlib.Path(checkpoint_dir) / "assets" / _checkpoints.TRAIN_SHAPE_FILE
    if _shape_file.exists():
        trained = json.loads(_shape_file.read_text())
        # Backward compat: checkpoints saved before the phase_tracking -> event_tracking
        # rename have this field recorded under the old name.
        if "phase_tracking" in trained:
            trained["event_tracking"] = trained.pop("phase_tracking")
        serving = _checkpoints._shape_signature(train_config.model)  # noqa: SLF001
        diffs = {k: (v, serving.get(k)) for k, v in trained.items() if serving.get(k) != v}
        if diffs:
            raise ValueError(
                f"Config '{train_config.name}' does not match how this checkpoint was trained: "
                + ", ".join(f"{k}: trained={t!r} serving={s!r}" for k, (t, s) in diffs.items())
                + f". Checkpoint: {checkpoint_dir}"
            )

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # NOTE: passing dtype=jnp.bfloat16 to restore_params() here throws for this
        # checkpoint format (root cause not fully diagnosed) — restore at the
        # checkpoint's native dtype instead.
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params"))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
