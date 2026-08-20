"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.xarm_policy as xarm_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional list of LeRobot repo ids. When set, the data loader concatenates these
    # datasets (each wrapped with its own prompt-from-task transform) and ignores ``repo_id``.
    # Use ``asset_id`` (via ``AssetsConfig``) to choose where the combined norm stats are stored.
    repo_ids: tuple[str, ...] | None = None
    # Optional per-repo sampling weights for multi-dataset training. Length must match ``repo_ids``.
    # If ``repo_ids`` is set and this is ``None``, sampling is balanced so that each repo is drawn
    # with equal probability regardless of its frame count. Pass explicit weights to override (the
    # values are normalized internally; e.g. ``(2.0, 1.0, 1.0)`` makes the first repo 2x more likely).
    # Ignored when only a single repo is in use.
    repo_weights: tuple[float, ...] | None = None
    # Optional per-repo episode exclusion, mapping repo_id -> episode indices to drop
    # at load time (non-destructive; the on-disk dataset is untouched). Use for
    # corrupted/inconsistent demos, e.g. {"lars/mem7": (0, 2, 3)} to drop the three
    # mem7 demos that end still-holding the gripper. Only applies to LeRobot datasets.
    exclude_episodes: dict[str, tuple[int, ...]] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, append `stop_padding_frames` zero-action "stop" frames at the end of each
    # episode (and zero-clamp the action-chunk tail for real frames near episode end) via
    # ActionPaddingWrapper. Defaults to false; enabled for the simulation training configs.
    stop_padding: bool = False
    # Number of zero-action stop frames appended per episode when `stop_padding` is true.
    stop_padding_frames: int = 50

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # Video encoder: raw LeRobot dataset feature keys whose temporal history
    # should be loaded (e.g. ("image", "wrist_image") for LIBERO).
    # Empty tuple = single-frame (non-video) behaviour.
    video_image_keys: tuple[str, ...] = ()
    # Time offsets in seconds for each video frame relative to the current
    # timestep. Negative = past. Length must equal model.num_frames.
    # e.g. (-3.0, -2.0, -1.0, 0.0) → 4 frames at 1 fps, 3 s of history.
    video_frame_offsets: tuple[float, ...] | None = None

    # Event memory training: instead of fixed temporal strides, load past frames
    # from the episode timesteps where a new event first appeared — mirroring
    # the event_memory inference behaviour.  Requires the dataset to have a
    # `labels` field with -1 for null frames and non-negative integers for events.
    # When True, `video_image_keys` names the keys to stack; `video_frame_offsets`
    # is ignored.
    event_memory_training: bool = False
    # Number of event-frame slots (must equal model.num_frames - 1).
    event_memory_size: int = 0
    # Probability of zeroing out all event frames for a given sample during training,
    # forcing the model to act without memory.  0.0 = always use event frames.
    event_dropout_prob: float = 0.0
    # Probability of replacing `phase_history` with "History: none" for a given sample
    # during training, independent of the event (video) dropout above.  Prevents the
    # model from becoming reliant on always having both memory modalities together,
    # which otherwise makes single-modality ablations at inference fail even when the
    # modality that IS present should be sufficient.  0.0 = always use real text.
    text_dropout_prob: float = 0.0
    # Number of frames at the start of each event transition to sample from during
    # training.  1 = always use the first labeled frame (old behaviour).  Setting
    # this to e.g. 10 randomly draws from the first 10 frames of the event, which
    # better matches the distribution of when the event detector fires at rollout
    # (typically several steps into the event rather than exactly at frame 0).
    event_frame_window: int = 1
    # Event ids whose FIRST occurrence is dropped from event memory. mem10 demos pour
    # twice into cup 1 (scoop→A, scoop→A, scoop→B) but the task is run one scoop per
    # cup at inference; skipping the first scoop/pour pair makes the training history
    # read (scoop,pour)→cup1, (scoop,pour)→cup2, matching what inference sees.
    # Empty = keep every occurrence.
    skip_first_event_ids: tuple[int, ...] = ()

    # Event-window upsampling: boosts the sampling weight of the ``upsample_window_steps``
    # frames immediately following the last labeled frame of event ``upsample_after_event_id``
    # in each episode, by a factor of ``upsample_weight``. Useful for a short "decision"
    # window (e.g. committing to move left vs. right after retracting) that's a small
    # fraction of the episode and would otherwise be undertrained. None = disabled.
    upsample_after_event_id: int | None = None
    upsample_window_steps: int = 0
    upsample_weight: float = 1.0
    # Frames to skip between the end of the event run and the start of the boosted window.
    # The stretch right after the event is often pre-decision (for mem10 the arm is still
    # leaving the bowl and the target isn't yet recoverable from the trajectory), so
    # boosting it spends weight on frames that teach nothing about the choice.
    upsample_window_offset: int = 0
    # Length of the state-noise window, when it should be SHORTER than the upsample
    # window. The two want different extents: noise belongs where the proprioceptive
    # shortcut lives (while the arm is committing), but must stop before the final
    # approach, where accurate position is what lets the arm actually hit the cup.
    # 0 = fall back to upsample_window_steps.
    noise_window_steps: int = 0
    # If true (and the upsample window above is configured), also add Gaussian noise to
    # `state` for frames inside that same window, so the policy can't use proprioceptive
    # drift as a shortcut for a decision meant to come from vision/phase_history memory.
    noise_state_in_upsample_window: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Optional list of LeRobot repo ids for multi-dataset training. When set, the
    # ``repo_id`` field is treated only as a fallback / display name, and you should
    # provide an explicit ``AssetsConfig.asset_id`` to name the combined norm-stats slot.
    repo_ids: tuple[str, ...] | None = None
    # Optional per-repo sampling weights (see ``DataConfig.repo_weights``).
    repo_weights: tuple[float, ...] | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            repo_ids=tuple(self.repo_ids) if self.repo_ids is not None else None,
            repo_weights=tuple(self.repo_weights) if self.repo_weights is not None else None,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.

    Video (video_encoder=True here) never combines with event tracking — this config
    has no `labels`/`phase_history` wiring at all. "Video" means a fixed time stride of
    past frames ending at the current one (see `frame_stride_sec` below); it trains the
    same video_encoder=True SigLIP temporal attention as
    `LeRobotLiberoEventKeyframeDataConfig`, just with a simpler, event-agnostic
    frame-selection pipeline and no auxiliary event-classification loss. For
    event_tracking=True, use `LeRobotLiberoEventDataConfig` (single frame) or
    `LeRobotLiberoEventKeyframeDataConfig` (video via actual past event frames) instead.
    """

    extra_delta_transform: bool = False
    # Seconds between consecutive video frames when video_encoder=True.
    # 1.0 s at 10 fps = every 10th frame; gives 3 s of history for num_frames=4.
    frame_stride_sec: float = 1.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # Video encoder: load T past frames via LeRobot delta_timestamps.
        video_image_keys: tuple[str, ...] = ()
        video_frame_offsets: tuple[float, ...] | None = None
        if getattr(model_config, "video_encoder", False):
            num_frames: int = model_config.num_frames  # type: ignore[attr-defined]
            stride = self.frame_stride_sec
            video_frame_offsets = tuple(-(num_frames - 1 - i) * stride for i in range(num_frames))
            video_image_keys = ("image", "wrist_image")

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            video_image_keys=video_image_keys,
            video_frame_offsets=video_frame_offsets,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotLiberoEventDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.

    For event_tracking=True models with video_encoder=False only (single current frame,
    no temporal stack). NOT interchangeable with `LeRobotLiberoEventKeyframeDataConfig`
    below: that class unconditionally turns on event_memory_training, which makes the
    data loader always emit a (T, C, H, W) stack — the wrong shape for a
    video_encoder=False model, which expects a plain (C, H, W) image. Video (plain
    `LeRobotLiberoDataConfig` above) never combines with event tracking — only this
    single-frame config and the keyframe config below load `labels`/`phase_history`.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "phase_history": "phase_history",
                        "labels": "labels",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotLiberoEventKeyframeDataConfig(DataConfigFactory):
    """Libero data config combining event tracking labels with event-triggered keyframes.

    "Keyframe" here means WHICH frames get fed to the video encoder: instead of a fixed
    time stride, each training sample's T-1 history frames are the actual past
    event-transition frames in that episode (via EventMemoryDataset), mirroring what
    `Policy`'s hidden-state cache serves at inference. The model side is identical to any
    other video_encoder=True model — same SigLIP temporal attention (see
    models/siglip.py's TemporalStrideBlock/VideoEncoder) — only the frame-selection
    pipeline differs. `LeRobotLiberoDataConfig` (`frame_stride_sec`) trains the same
    model via a fixed-stride ("naive video") pipeline, but never combines with event
    tracking — only this class and `LeRobotLiberoEventDataConfig` (single frame, no
    video) load `labels`/`phase_history`.

    Used for event-memory models trained with both video_encoder=True and
    event_tracking=True.  Provides per-sample event labels for the classification
    loss AND assembles the video stack from the actual event-transition frames in
    each episode (via EventMemoryDataset), mirroring the inference behaviour.
    """

    extra_delta_transform: bool = False
    event_dropout_prob: float = 0.0
    text_dropout_prob: float = 0.0
    event_frame_window: int = 1
    upsample_after_event_id: int | None = None
    upsample_window_steps: int = 0
    upsample_weight: float = 1.0
    noise_state_in_upsample_window: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "phase_history": "phase_history",
                        "labels": "labels",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        num_frames: int = model_config.num_frames  # type: ignore[attr-defined]

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            video_image_keys=("image", "wrist_image"),
            event_memory_training=True,
            event_memory_size=num_frames - 1,
            event_dropout_prob=self.event_dropout_prob,
            text_dropout_prob=self.text_dropout_prob,
            event_frame_window=self.event_frame_window,
            upsample_after_event_id=self.upsample_after_event_id,
            upsample_window_steps=self.upsample_window_steps,
            upsample_weight=self.upsample_weight,
            noise_state_in_upsample_window=self.noise_state_in_upsample_window,
        )

@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotXarmEventDataConfig(DataConfigFactory):
    """
    Example data config for custom Xarm dataset in LeRobot format.

    For event_tracking=True models with video_encoder=False only (single current frame,
    no temporal stack). NOT interchangeable with `LeRobotXarmEventKeyframeDataConfig`:
    that class unconditionally turns on event_memory_training, which makes the data
    loader always emit a (T, C, H, W) stack — the wrong shape for a video_encoder=False
    model, which expects a plain (C, H, W) image (same reasoning as
    `LeRobotLiberoEventDataConfig` vs. `LeRobotLiberoEventKeyframeDataConfig` above).
    Note `LeRobotXarmVideoDataConfig` below is the video counterpart with NO event
    tracking at all — video never combines with events, only keyframes do.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/state": "state",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                        "phase_history": "phase_history",
                        "labels": "labels",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[xarm_policy.XarmInputs(model_type=model_config.model_type)],
            outputs=[xarm_policy.XarmOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotXarmVideoDataConfig(DataConfigFactory):
    """Xarm data config with video encoder support (no event labels)."""

    # Seconds between consecutive video frames (2.0 s × 20 fps = every 40 steps).
    frame_stride_sec: float = 2.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/state": "state",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[xarm_policy.XarmInputs(model_type=model_config.model_type)],
            outputs=[xarm_policy.XarmOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        video_image_keys: tuple[str, ...] = ()
        video_frame_offsets: tuple[float, ...] | None = None
        if getattr(model_config, "video_encoder", False):
            num_frames: int = model_config.num_frames  # type: ignore[attr-defined]
            stride = self.frame_stride_sec
            video_frame_offsets = tuple(-(num_frames - 1 - i) * stride for i in range(num_frames))
            video_image_keys = ("exterior_image_1_left", "wrist_image_left")

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            video_image_keys=video_image_keys,
            video_frame_offsets=video_frame_offsets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotXarmEventKeyframeDataConfig(DataConfigFactory):
    """Xarm data config combining event tracking labels with event-triggered keyframes.

    "Keyframe" here means WHICH frames get fed to the video encoder: instead of a fixed
    time stride, each training sample's T-1 history frames are the actual past
    event-transition frames in that episode (via EventMemoryDataset), mirroring what
    `Policy`'s hidden-state cache serves at inference. The model side is identical to any
    other video_encoder=True model — same SigLIP temporal attention (see
    models/siglip.py's TemporalStrideBlock/VideoEncoder) — only the frame-selection
    pipeline differs. See `LeRobotXarmVideoDataConfig` for the fixed-stride ("naive
    video") counterpart (without event tracking).

    Used for event-memory models trained with both video_encoder=True and
    event_tracking=True.  Provides per-sample event labels for the classification
    loss AND assembles the video stack from the actual event-transition frames in
    each episode (via EventMemoryDataset), mirroring the inference behaviour.
    """

    event_dropout_prob: float = 0.0
    text_dropout_prob: float = 0.0
    event_frame_window: int = 1
    skip_first_event_ids: tuple[int, ...] = ()
    upsample_after_event_id: int | None = None
    upsample_window_steps: int = 0
    upsample_weight: float = 1.0
    upsample_window_offset: int = 0
    noise_window_steps: int = 0
    noise_state_in_upsample_window: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/state": "state",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                        "phase_history": "phase_history",
                        "labels": "labels",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[xarm_policy.XarmInputs(model_type=model_config.model_type)],
            outputs=[xarm_policy.XarmOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        num_frames: int = model_config.num_frames  # type: ignore[attr-defined]

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            video_image_keys=("exterior_image_1_left", "wrist_image_left"),
            event_memory_training=True,
            event_memory_size=num_frames - 1,
            event_dropout_prob=self.event_dropout_prob,
            text_dropout_prob=self.text_dropout_prob,
            event_frame_window=self.event_frame_window,
            skip_first_event_ids=self.skip_first_event_ids,
            upsample_after_event_id=self.upsample_after_event_id,
            upsample_window_steps=self.upsample_window_steps,
            upsample_weight=self.upsample_weight,
            upsample_window_offset=self.upsample_window_offset,
            noise_window_steps=self.noise_window_steps,
            noise_state_in_upsample_window=self.noise_state_in_upsample_window,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    # Multiplier on the LR schedule for ``Pi0.phase_head`` only (JAX training). Backbone uses the base schedule.
    # 1.0 disables per-module LR. Values >1 train the phase_head classifier faster than the PaliGemma backbone.
    phase_head_lr_multiplier: float = 1.0
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # bnb for pytorch models
    bnb: bool = False

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=56,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ============================================================================
    # UniMem example configs.
    #
    # These three show how to add event tracking (an auxiliary classifier that
    # detects semantic events like "grabbed box" or "tapped left basket") and,
    # optionally, an event-conditioned video encoder, to your own LeRobot dataset.
    # Every real experiment config below (libero_mem*, xarm_mem*) is a variation on
    # this same shape — copy whichever of these three is closest to your setup and
    # start pruning/tuning from there. See the README's "Training your own
    # event-memory policy" section for the full walkthrough, and
    # examples/libero/label_dataset_libero.py / examples/xarm/label_dataset_xarm.py for
    # how to generate the `labels` (event id) and `phase_history` (text summary) columns
    # these configs expect your dataset to have.
    # ============================================================================
    #
    TrainConfig(
        # Example 1: single-frame event tracking, no video encoder. This is the
        # simplest way to add event detection to an existing Pi0.5 fine-tune — the
        # model still only ever sees the CURRENT frame; it just also learns to
        # classify which semantic event (if any) is happening in it. Start here, then
        # move to `unimem_example_libero_keyframe` once single-frame event detection
        # works and you want the policy to actually condition its actions on past
        # events, not just detect them. (`unimem_example_libero_video` is a separate,
        # event-agnostic baseline — video never combines with event tracking; only
        # keyframes do — see that config's comment below.)
        name="unimem_example_libero",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            # LoRA fine-tuning fits on a single >22.5GB GPU (see README's Requirements
            # table). Drop "_lora" from both variants for a full fine-tune (>70GB).
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Adds Pi0.phase_head (an MLP classifier on the pooled prefix) and its
            # auxiliary cross-entropy loss (see Pi0.compute_loss_event). Requires your
            # dataset to have an integer `labels` column: -1 for unlabeled/no-event
            # frames, 0..N-1 for your N event classes.
            event_tracking=True,
        ),
        data=LeRobotLiberoEventDataConfig(
            # Your LeRobot dataset's repo id (HuggingFace hub id, or a local path).
            repo_id="your_hf_username/my_event_dataset",
            # Give it its own asset id so its norm stats don't collide with anything
            # else in ./assets. Defaults to repo_id if omitted.
            assets=AssetsConfig(asset_id="my_event_dataset"),
            base_config=DataConfig(
                prompt_from_task=True,
                # Zero-action "stop" frames appended per episode so the policy learns
                # to come to rest instead of trailing off. Helpful on real robots;
                # less critical in sim (e.g. LIBERO).
                stop_padding=True,
            ),
        ),
        # Multiplier on the LR schedule below, applied to phase_head params only
        # (backbone/LoRA still use the base schedule). The head is a small MLP
        # trained from scratch, so bumping this above 1.0 can help it catch up to an
        # already-pretrained backbone; 1.0 (shown here) is a safe starting point —
        # every one of our own experiments below also leaves it at 1.0.
        phase_head_lr_multiplier=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # Freezes everything except the LoRA adapters and phase_head (repeat the same
        # model config so the filter's path regexes match what was just built above).
        # Drop this line entirely for a full fine-tune.
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # EMA defaults to 0.99; we disable it for LoRA fine-tunes since the adapter
        # weights are small and converge quickly enough that EMA mostly just delays
        # convergence. Re-enable (remove this line) for a full fine-tune.
        ema_decay=None,
        batch_size=32,
        num_train_steps=20_000,
        log_interval=20,
        save_interval=1_000,
        keep_period=10_000,
    ),
    TrainConfig(
        # Example 2: same idea as `unimem_example_libero` above, but for a real-robot
        # (xArm) dataset instead of a sim one. The main differences from the LIBERO
        # example are the data config class (XarmInputs/XarmOutputs instead of
        # LiberoInputs/LiberoOutputs — see src/openpi/policies/xarm_policy.py) and
        # `text_dropout_prob`, which we found necessary on real hardware to keep the
        # policy from over-relying on the text event-history summary.
        name="unimem_example_xarm",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="your_hf_username/my_xarm_event_dataset",
            assets=AssetsConfig(asset_id="my_xarm_event_dataset"),
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
                # Randomly replace `phase_history` with "History: none" for this
                # fraction of training samples, independent of any video/event-frame
                # dropout. Prevents the policy from becoming reliant on the text
                # summary always being informative, which otherwise makes ablations
                # that drop it at inference fail even when vision alone should
                # suffice. 0.0 = always use the real text.
                text_dropout_prob=0.2,
            ),
        ),
        phase_head_lr_multiplier=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000, decay_lr=3.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        ema_decay=None,
        batch_size=32,
        num_train_steps=20_000,
        log_interval=20,
        save_interval=1_000,
        keep_period=10_000,
    ),
    TrainConfig(
        # Example 3: adds the video encoder on top of `unimem_example_libero`, so the
        # policy conditions on past EVENT frames (not just the current one) via
        # temporal attention in SigLIP — see src/openpi/models/siglip.py's
        # TemporalStrideBlock/VideoEncoder and models/siglip_hidden_cache.py for how
        # this is served efficiently at inference time. This is the full "event
        # memory" setup our paper's checkpoints use.
        #
        # "Keyframe" here is about the DATA PIPELINE, not the model: video_encoder=True
        # always feeds the same SigLIP temporal-attention stack a stack of T frames —
        # this config just chooses those T frames as the actual past EVENT frames
        # (via EventMemoryDataset) instead of a fixed time stride. Compare against
        # `unimem_example_libero_video` further below, which uses the identical
        # video_encoder=True architecture with a fixed-stride pipeline instead — but
        # unlike this one, that example has NO event tracking at all (video never
        # combines with events; only keyframes do).
        name="unimem_example_libero_keyframe",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            # T frames per camera: the current frame plus (T-1) past EVENT frames
            # (not a fixed time window — see EventMemoryDataset in
            # training/data_loader.py). Must match the served config exactly (see
            # policy_config.py's train_shape.json guard) and determines the
            # hidden-state cache depth at inference (siglip_hidden_cache.py).
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            repo_id="your_hf_username/my_event_dataset",
            assets=AssetsConfig(asset_id="my_event_dataset"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            # Probability of zeroing out all event frames for a sample, forcing the
            # model to act from the current frame + text history alone. 0.0 = never.
            event_dropout_prob=0.2,
            # Probability of replacing `phase_history` with "History: none",
            # independent of the video dropout above — see the xarm example's
            # comment on `text_dropout_prob` for why both are needed together.
            text_dropout_prob=0.2,
            # Number of frames at the start of each event to sample from, instead of
            # always the first labeled frame. Matches the variability in when your
            # event detector actually fires at rollout (e.g. a few steps into the
            # event rather than exactly at frame 0). 1 = old/deterministic behaviour.
            event_frame_window=30,
            # Optional: boost the sampling weight of frames shortly after a chosen
            # event id, for a short "decision window" that would otherwise be an
            # undertrained sliver of the episode (e.g. a fork in the task right after
            # a pickup). None = disabled. See training/config.py's DataConfig fields
            # (upsample_window_steps, upsample_weight, upsample_window_offset) for the
            # rest of this knob, and LeRobotXarmEventKeyframeDataConfig /
            # `skip_first_event_ids` for a fancier dedup-first-occurrence variant we
            # needed for one of our xArm tasks (a repeated event within one episode).
            upsample_after_event_id=None,
        ),
        phase_head_lr_multiplier=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        ema_decay=None,
        # Video training holds num_frames x as many images in memory per sample, so
        # this typically needs a smaller batch_size than the single-frame examples
        # above for the same GPU — tune down from here if you hit OOMs.
        batch_size=24,
        num_train_steps=20_000,
        log_interval=20,
        save_interval=1_000,
        keep_period=10_000,
    ),
    TrainConfig(
        # Example 4: adds the SAME video_encoder=True SigLIP temporal attention as
        # `unimem_example_libero_keyframe`, but with NO event tracking at all — no
        # phase_head, no `labels`/`phase_history`, no auxiliary loss. Video never
        # combines with event tracking; only keyframe sampling does (see
        # `LeRobotLiberoDataConfig`'s docstring). This is a plain "does temporal
        # context help the policy at all" baseline, using a fixed `frame_stride_sec`
        # instead of event-triggered frames — useful to compare against
        # `unimem_example_libero` (no history) before you bring in event tracking at
        # all via the keyframe example.
        name="unimem_example_libero_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/my_video_dataset",
            assets=AssetsConfig(asset_id="my_video_dataset"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            # Seconds between consecutive history frames, ending at the current one.
            # 1.0s at 10fps = every 10th frame; with num_frames=3 that's 2s of history.
            frame_stride_sec=1.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=3,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        ema_decay=None,
        batch_size=24,
        num_train_steps=20_000,
        log_interval=20,
        save_interval=1_000,
        keep_period=10_000,
    ),
    #
    # ============================================================================
    # End of UniMem example configs. Everything below is the real experiment
    # sweep used to produce our paper's checkpoints (LIBERO sim + real xArm),
    # kept as-is for reproducibility.
    # ============================================================================
    #
    TrainConfig(
        name="libero_mem1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            assets=AssetsConfig(asset_id="lars/mem1"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
        ),
    ),
    TrainConfig(
        name="libero_mem2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            assets=AssetsConfig(asset_id="lars/mem2"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
        ),
    ),
    TrainConfig(
        name="libero_mem3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            assets=AssetsConfig(asset_id="lars/mem3"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
        ),
    ),
    TrainConfig(
        name="libero_mem4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            assets=AssetsConfig(asset_id="lars/mem4"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
        ),
    ),
    TrainConfig(
        name="libero_mem5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            assets=AssetsConfig(asset_id="lars/mem5"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
        ),
    ),
    TrainConfig(
        name="libero_mem6",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            assets=AssetsConfig(asset_id="lars/mem6"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
        ),
    ),
    TrainConfig(
        name="xarm_mem7_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(asset_id="lars/mem7"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem8_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(asset_id="lars/mem8"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem9_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem9_v3",
            assets=AssetsConfig(asset_id="lars/mem9"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem10_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            # MUST match xarm_mem10_coruscant (the checkpoint this config serves).
            # num_frames sets both the hidden-state cache depth (num_frames-1 history slots) and
            # the temporal PE table length; a mismatch loads and runs without error but
            # silently serves the model with a different number of history slots than it
            # was trained on. Keep these two in lockstep whenever either changes.
            num_frames=4,
            event_tracking=True,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            # Match xarm_mem10_coruscant, which trains on the trimmed set. Serving reads
            # norm stats out of the checkpoint's own assets dir, so this is mostly
            # bookkeeping — but a repo_id that names a different dataset than the one the
            # weights were fit to is exactly how the 6-vs-8 num_frames mismatch hid.
            repo_id="lars/mem10_v4_final",
            assets=AssetsConfig(asset_id="lars/mem10"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    # ------------------------------------------------------------------------------
    # Inference configs for the mem7-mem10 memory ablations.  One per training arm; the
    # model block MUST mirror the arm it serves (see the num_frames note on
    # xarm_mem10_infer — a mismatch loads and runs silently).
    #
    # The `_no_memory` / `_text_only` arms have NO video encoder, so serve them with
    # `--video_encoder=False` in examples/xarm/xarm_inference.py: that skips the stateful
    # keyframe hidden-state cache path, while `_feed_text_history` still sends a `phase_history`
    # string in every event-tracking mode (pinned to "History: none" for no_memory).
    # Suggested rollout mode per arm: _no_memory -> mode=no_memory, _text_only ->
    # mode=text, _keyframe_only -> mode=keyframe (with --video_encoder).
    #
    # asset_id must match what the corresponding training config wrote into the
    # checkpoint's assets dir; norm stats are read from there, not from ./assets.
    TrainConfig(
        name="xarm_mem7_no_memory_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(asset_id="lars/mem7"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem7_text_only_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(asset_id="lars/mem7"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem7_keyframe_only_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(asset_id="lars/mem7"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem8_no_memory_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(asset_id="lars/mem8"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem8_text_only_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(asset_id="lars/mem8"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem8_keyframe_only_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(asset_id="lars/mem8"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem9_no_memory_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem9_v3",
            # Matches the asset_id the mem9 ablation configs train with.
            assets=AssetsConfig(asset_id="lars/mem9"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem9_text_only_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem9_v3",
            assets=AssetsConfig(asset_id="lars/mem9"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem9_keyframe_only_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem9_v3",
            assets=AssetsConfig(asset_id="lars/mem9"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem10_no_memory_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem10_v4_final",
            assets=AssetsConfig(asset_id="lars/mem10"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem10_text_only_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem10_v4_final",
            assets=AssetsConfig(asset_id="lars/mem10"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    TrainConfig(
        name="xarm_mem10_keyframe_only_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem10_v4_final",
            assets=AssetsConfig(asset_id="lars/mem10"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
    ),
    # video inference configs: same architecture as video finetune
    TrainConfig(
        name="libero_mem1_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            assets=AssetsConfig(asset_id="lars/mem1"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
    ),
    TrainConfig(
        name="libero_mem2_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            assets=AssetsConfig(asset_id="lars/mem2"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
    ),
    TrainConfig(
        name="libero_mem3_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            assets=AssetsConfig(asset_id="lars/mem3"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
    ),
    TrainConfig(
        name="libero_mem4_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            assets=AssetsConfig(asset_id="lars/mem4"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
    ),
    TrainConfig(
        name="libero_mem5_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            assets=AssetsConfig(asset_id="lars/mem5"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
    ),
    TrainConfig(
        name="libero_mem6_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            assets=AssetsConfig(asset_id="lars/mem6"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
    ),
    TrainConfig(
        name="xarm_mem7_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmVideoDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(asset_id="lars/mem7"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            frame_stride_sec=6.0,
        ),
    ),
    TrainConfig(
        name="xarm_mem8_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmVideoDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(asset_id="lars/mem8"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            frame_stride_sec=6.0,
        ),
    ),
    TrainConfig(
        name="xarm_mem9_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmVideoDataConfig(
            repo_id="lars/mem9_v3",
            assets=AssetsConfig(asset_id="lars/mem9"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            frame_stride_sec=6.0,
        ),
    ),
    TrainConfig(
        name="xarm_mem10_video_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            # MUST match xarm_mem10_video (the checkpoint this serves), which is itself
            # frame-matched to xarm_mem10_coruscant.
            num_frames=6,
        ),
        data=LeRobotXarmVideoDataConfig(
            repo_id="lars/mem10",
            assets=AssetsConfig(asset_id="lars/mem10"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            frame_stride_sec=6.0,
        ),
    ),
    #
    # All finetuning configs for our experiments (ignore coruscant vs finetune, they're the same)
    #
    TrainConfig(
        name="libero_mem1_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            repo_id="lars/mem1",
            assets=AssetsConfig(asset_id="lars/mem1"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            event_dropout_prob=0.5,
            text_dropout_prob=0.5,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=2.5e-6,       
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=24,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="libero_mem2_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            repo_id="lars/mem2_3",
            assets=AssetsConfig(asset_id="lars/mem2"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            event_dropout_prob=0.5,
            text_dropout_prob=0.5,
            upsample_after_event_id=6,
            upsample_window_steps=25,
            upsample_weight=3.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,
            decay_lr=2.5e-6,       
        ),
        log_interval=20,
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=24,
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="libero_mem3_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            repo_id="lars/mem3_left",
            repo_ids=("lars/mem3_left", "lars/mem3_right"),
            assets=AssetsConfig(asset_id="lars/mem3"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            event_dropout_prob=0.5,
            text_dropout_prob=0.5,
            upsample_after_event_id=3,
            upsample_window_steps=90,
            upsample_weight=5.0,
            noise_state_in_upsample_window=True,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=2.5e-6,       
        ),
        log_interval=20,
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=24,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="libero_mem4_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=3,
            event_tracking=True,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            repo_id="lars/mem4",
            assets=AssetsConfig(asset_id="lars/mem4"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            event_dropout_prob=0.5,
            text_dropout_prob=0.5,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=2.5e-6,       
        ),
        log_interval=20,
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=3,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=24,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
    ),
    TrainConfig(
        name="libero_mem5_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            repo_id="lars/mem5_1",
            repo_ids=("lars/mem5_1", "lars/mem5_2"),
            assets=AssetsConfig(asset_id="lars/mem5"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            event_dropout_prob=0.5,
            text_dropout_prob=0.5,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=2.5e-6,       
        ),
        log_interval=20,
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=3,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=24,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="libero_mem6_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=3,
            event_tracking=True,
        ),
        data=LeRobotLiberoEventKeyframeDataConfig(
            repo_id="lars/mem6_1",
            repo_ids=("lars/mem6_1", "lars/mem6_2", "lars/mem6_3", "lars/mem6_4"),
            assets=AssetsConfig(asset_id="lars/mem6"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            event_dropout_prob=0.5,
            text_dropout_prob=0.5,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,  
            decay_lr=3.5e-6,       
        ),
        log_interval=20,
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=3,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=24,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem7_coruscant",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(asset_id="lars/mem7"),
            # Drop the 3 mem7 demos that end still-holding the gripper.
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
            ),
            event_dropout_prob=0.0,
            text_dropout_prob=0.0,
            event_frame_window=30,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="xarm_mem8_coruscant",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(asset_id="lars/mem8"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            event_dropout_prob=0.0,
            text_dropout_prob=0.0,
            event_frame_window=30,
            upsample_after_event_id=2,
            upsample_window_steps=160,
            upsample_weight=5.0,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,  
            decay_lr=3.5e-6,       
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem9_coruscant",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,             
            event_tracking=True,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem9_v3",
            assets=AssetsConfig(asset_id="lars/mem9_v3"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            event_dropout_prob=0.0,
            text_dropout_prob=0.0,
            event_frame_window=25,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,  
            decay_lr=3.5e-6,       
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem10_coruscant",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem10_v4_final",
            repo_ids=("lars/mem10_v4_final", "lars/mem10_v5_final_relabel"),
            assets=AssetsConfig(asset_id="lars/mem10"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            event_dropout_prob=0.0,
            text_dropout_prob=0.0,
            event_frame_window=40,
            upsample_after_event_id=2,  # scooped beans
            upsample_window_offset=-36,
            upsample_window_steps=76,
            upsample_weight=4.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=40_000,  
            peak_lr=2.5e-5,
            decay_lr=3.5e-6,       
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
    ),
    # ------------------------------------------------------------------------------
    # Memory ablations for mem7-mem10.  Three arms per task, each identical to that
    # task's `xarm_memN_coruscant` config (same data, schedule, batch size, upsampling,
    # freeze filter) except for which memory modality reaches the policy:
    #
    #   _no_memory      neither modality. video_encoder=False, phase_history pinned to
    #                   "History: none".
    #   _text_only      textual memory only. video_encoder=False, real phase_history.
    #   _keyframe_only  keyframe memory only. Event-memory video stack as usual, but
    #                   phase_history pinned to "History: none" (text_dropout_prob=1.0).
    #
    # The two no-keyframe arms drop the video encoder entirely (num_frames=1) rather
    # than feeding zeroed keyframes, so they cost one SigLIP forward per sample instead
    # of four.
    #
    # INVARIANT — the event target loss is IDENTICAL for every arm AND for the
    # `xarm_memN_coruscant` parent: `event_tracking=True`, `phase_head_lr_multiplier=1.0`,
    # `labels` mapped into the batch, and the phase_head left unfrozen (the LoRA freeze
    # filters only catch `llm` params).  The objective's internals — the 0.1 loss weight,
    # the -1 -> ignore-class mapping, the 0.02 downweight on unlabeled frames — are
    # hardcoded in `Pi0.compute_loss_event`, so they cannot drift per config.
    #
    # Keeping it on in every arm is deliberate, not incidental.  The phase_head reads the
    # CURRENT frame and the model has no recurrence, so the auxiliary loss cannot leak
    # memory into a no-memory arm; occurrences of a repeated event share one label id
    # (e.g. mem8 runs 0,1,2,1,2,1,2,3), so the target does not encode "which repetition".
    # Turning it off for `_no_memory` alone would make that arm differ from `_text_only`
    # in TWO ways at once, and `_text_only` needs the head regardless — the detector is
    # what builds the history string at rollout.
    #
    # `phase_history` is always present in the prompt, pinned to "History: none" when the
    # arm has no textual memory.  Dropping the key instead would change the tokenizer's
    # output format (`Task: …, State: …` vs `Task: …, {events}, State: …`) and confound
    # the ablation with a prompt-shape change.
    #
    # Norm stats are read from each task's existing `xarm_memN_coruscant` assets dir, so
    # all four arms of a task share one normalization (and no recompute is needed).
    TrainConfig(
        name="xarm_mem7_no_memory",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(assets_dir="./assets/xarm_mem7_coruscant", asset_id="lars/mem7"),
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
                text_dropout_prob=1.0,
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,
            decay_lr=3.5e-6,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ).get_freeze_filter(),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="xarm_mem7_text_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(assets_dir="./assets/xarm_mem7_coruscant", asset_id="lars/mem7"),
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="xarm_mem7_keyframe_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(assets_dir="./assets/xarm_mem7_coruscant", asset_id="lars/mem7"),
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
            ),
            event_dropout_prob=0.0,
            text_dropout_prob=1.0,
            event_frame_window=30,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="xarm_mem8_no_memory",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(assets_dir="./assets/xarm_mem8_coruscant", asset_id="lars/mem8"),
            # Upsampling is applied from the DataConfig by the sampler (it only needs
            # `labels`), so it works the same on this non-video path.
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
                text_dropout_prob=1.0,
                upsample_after_event_id=2,
                upsample_window_steps=160,
                upsample_weight=5.0,
            ),
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem8_text_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(assets_dir="./assets/xarm_mem8_coruscant", asset_id="lars/mem8"),
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
                upsample_after_event_id=2,
                upsample_window_steps=160,
                upsample_weight=5.0,
            ),
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=False,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem8_keyframe_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(assets_dir="./assets/xarm_mem8_coruscant", asset_id="lars/mem8"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            event_dropout_prob=0.0,
            text_dropout_prob=1.0,
            event_frame_window=30,
            upsample_after_event_id=2,
            upsample_window_steps=160,
            upsample_weight=5.0,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            event_tracking=True,
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem9_no_memory",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem9_v3",
            # NOTE: asset_id is "lars/mem9" (not "lars/mem9_v3") because that is where the
            # mem9 norm stats actually live on disk and in the mem9 checkpoint.
            assets=AssetsConfig(assets_dir="./assets/xarm_mem9_coruscant", asset_id="lars/mem9"),
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
                text_dropout_prob=1.0,
            ),
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem9_text_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem9_v3",
            # NOTE: asset_id is "lars/mem9" (not "lars/mem9_v3") because that is where the
            # mem9 norm stats actually live on disk and in the mem9 checkpoint.
            assets=AssetsConfig(assets_dir="./assets/xarm_mem9_coruscant", asset_id="lars/mem9"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem9_keyframe_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem9_v3",
            # NOTE: asset_id is "lars/mem9" (not "lars/mem9_v3") because that is where the
            # mem9 norm stats actually live on disk and in the mem9 checkpoint.
            assets=AssetsConfig(assets_dir="./assets/xarm_mem9_coruscant", asset_id="lars/mem9"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            event_dropout_prob=0.0,
            text_dropout_prob=1.0,
            event_frame_window=25,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem10_no_memory",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem10_v4_final",
            repo_ids=("lars/mem10_v4_final", "lars/mem10_v5_final_relabel"),
            assets=AssetsConfig(assets_dir="./assets/xarm_mem10_coruscant", asset_id="lars/mem10"),
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
                text_dropout_prob=1.0,
                upsample_after_event_id=2,  # scooped beans
                upsample_window_offset=-36,
                upsample_window_steps=76,
                upsample_weight=4.0,
            ),
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=40_000,
            peak_lr=2.5e-5,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
    ),
    TrainConfig(
        name="xarm_mem10_text_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ),
        data=LeRobotXarmEventDataConfig(
            repo_id="lars/mem10_v4_final",
            repo_ids=("lars/mem10_v4_final", "lars/mem10_v5_final_relabel"),
            assets=AssetsConfig(assets_dir="./assets/xarm_mem10_coruscant", asset_id="lars/mem10"),
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
                upsample_after_event_id=2,  # scooped beans
                upsample_window_offset=-36,
                upsample_window_steps=76,
                upsample_weight=4.0,
            ),
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=40_000,
            peak_lr=2.5e-5,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=False,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
    ),
    TrainConfig(
        name="xarm_mem10_keyframe_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ),
        data=LeRobotXarmEventKeyframeDataConfig(
            repo_id="lars/mem10_v4_final",
            repo_ids=("lars/mem10_v4_final", "lars/mem10_v5_final_relabel"),
            assets=AssetsConfig(assets_dir="./assets/xarm_mem10_coruscant", asset_id="lars/mem10"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            event_dropout_prob=0.0,
            text_dropout_prob=1.0,
            event_frame_window=40,
            upsample_after_event_id=2,  # scooped beans
            upsample_window_offset=-36,
            upsample_window_steps=76,
            upsample_weight=4.0,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=40_000,
            peak_lr=2.5e-5,
            decay_lr=3.5e-6,
        ),
        log_interval=20,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
            event_tracking=True,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
    ),
    # Video (stride) finetune configs — video_encoder=True, event_tracking=False, fixed temporal stride.
    TrainConfig(
        name="libero_mem1_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="lars/mem1",
            assets=AssetsConfig(asset_id="lars/mem1"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=3.5e-6,       
        ),
        log_interval=20,
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=16,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="libero_mem2_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="lars/mem2_3",
            assets=AssetsConfig(asset_id="lars/mem2"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=3.5e-6,       
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=16,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="libero_mem3_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="lars/mem3_left",
            repo_ids=("lars/mem3_left", "lars/mem3_right"),
            assets=AssetsConfig(asset_id="lars/mem3"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=3.5e-6,       
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=16,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="libero_mem4_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="lars/mem4",
            assets=AssetsConfig(asset_id="lars/mem4"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=3.5e-6,       
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=16,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="libero_mem5_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="lars/mem5_1",
            repo_ids=("lars/mem5_1", "lars/mem5_2"),
            assets=AssetsConfig(asset_id="lars/mem5"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=3.5e-6,       
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=16,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="libero_mem6_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="lars/mem6_1",
            repo_ids=("lars/mem6_1", "lars/mem6_2", "lars/mem6_3", "lars/mem6_4"),
            assets=AssetsConfig(asset_id="lars/mem6"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=True),
            extra_delta_transform=False,
            frame_stride_sec=6.0,
        ),
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=3.5e-6,       
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=16,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="xarm_mem7_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmVideoDataConfig(
            repo_id="lars/mem7",
            assets=AssetsConfig(asset_id="lars/mem7"),
            # Drop the 3 mem7 demos that end still-holding the gripper (~0.5) instead of
            # releasing — contradictory supervision for "open at the end".
            base_config=DataConfig(
                prompt_from_task=True,
                stop_padding=False,
                exclude_episodes={"lars/mem7": (0, 2, 3)},
            ),
            frame_stride_sec=6.0,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=20_000,  
            decay_lr=3.5e-6,       
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="xarm_mem8_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmVideoDataConfig(
            repo_id="lars/mem8",
            assets=AssetsConfig(asset_id="lars/mem8"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            frame_stride_sec=6.0,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,  
            decay_lr=3.5e-6,       
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem9_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ),
        data=LeRobotXarmVideoDataConfig(
            repo_id="lars/mem9_v3",
            assets=AssetsConfig(asset_id="lars/mem9"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            frame_stride_sec=6.0,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,  
            decay_lr=3.5e-6,       
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=4,
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=44,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="xarm_mem10_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            # Frame-matched to xarm_mem10_coruscant so the video baseline differs only in
            # HOW frames are chosen (fixed stride vs event keyframes), not how many.
            num_frames=6,
        ),
        data=LeRobotXarmVideoDataConfig(
            repo_id="lars/mem10",
            assets=AssetsConfig(asset_id="lars/mem10"),
            base_config=DataConfig(prompt_from_task=True, stop_padding=False),
            frame_stride_sec=6.0,
        ),
        #checkpoint_base_dir="/path/to/your/checkpoints",
        log_interval=20,
        lr_schedule=_optimizer.CosineDecaySchedule(
            decay_steps=30_000,  
            decay_lr=3.5e-6,       
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            video_encoder=True,
            num_frames=6,  # keep in lockstep with the model config above
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=10_000,
        save_interval=1_000,
        batch_size=36,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/your/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
