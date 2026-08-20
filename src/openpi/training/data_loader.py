from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp

import openpi.training.lerobot_hf_patch  # noqa: F401  # patches LeRobot hf_transform_to_torch before dataset import
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
            # Event / classification targets (ignored unless model.event_tracking).
            "labels": np.int32(0),
        }

    def __len__(self) -> int:
        return self._num_samples


class EventMemoryDataset(torch.utils.data.Dataset):
    """Wraps a LeRobotDataset to assemble event-triggered video stacks for training.

    For each training sample at timestep t in episode e, this dataset:
      1. Scans the episode's event labels to find every timestep where a NEW event
         first appeared before t (mirroring xarm_inference.py event detection).
      2. Loads the image frames at those event timesteps.
      3. Returns the images stacked as (T, C, H, W) = [pads, event_frames, current],
         matching what _build_event_obs produces at inference.

    The base_dataset must be a LeRobotDataset whose delta_timestamps include
    the image keys at offset [0.0] (single current frame per key).  Event labels
    are read from the HuggingFace dataset's "labels" column, where -1 = null/unknown.
    """

    NULL_LABEL: int = -1

    def __init__(
        self,
        base_dataset: lerobot_dataset.LeRobotDataset,
        image_keys: tuple[str, ...],
        event_memory_size: int,
        dropout_prob: float = 0.0,
        text_dropout_prob: float = 0.0,
        event_frame_window: int = 1,
        skip_first_event_ids: tuple[int, ...] = (),
    ) -> None:
        self._base_dataset = base_dataset
        self._image_keys = image_keys
        self._event_memory_size = event_memory_size
        self._dropout_prob = dropout_prob
        self._text_dropout_prob = text_dropout_prob
        self._event_frame_window = event_frame_window
        self._skip_first_event_ids = tuple(skip_first_event_ids)

        hf_ds = base_dataset.hf_dataset
        episode_indices = np.array(hf_ds["episode_index"])
        labels_raw = hf_ds["labels"]
        labels = np.array(
            [int(l.item()) if hasattr(l, "item") else (int(l[0]) if hasattr(l, "__len__") else int(l)) for l in labels_raw],
            dtype=np.int32,
        )

        # For each global HF frame index, store a list of candidate-index lists
        # (one per event, each containing the first event_frame_window frame indices
        # of that event).  At __getitem__ time we randomly draw one candidate per event,
        # matching the distribution of when the event detector actually fires at rollout.
        self._event_frame_candidates: dict[int, list[list[int]]] = {}

        for ep_id in np.unique(episode_indices):
            ep_global = np.where(episode_indices == ep_id)[0]  # global HF indices
            ep_labels = labels[ep_global]

            # Build per-event candidate windows: first event_frame_window frames per event.
            # Each entry keeps its event id so occurrences can be filtered below.
            event_windows: list[tuple[list[int], int]] = []
            last_event = self.NULL_LABEL
            current_window: list[int] = []

            for gidx, lbl in zip(ep_global, ep_labels):
                if lbl != self.NULL_LABEL and lbl != last_event:
                    if current_window:
                        event_windows.append((current_window, last_event))
                    current_window = [int(gidx)]
                    last_event = int(lbl)
                elif lbl == last_event and lbl != self.NULL_LABEL and len(current_window) < event_frame_window:
                    current_window.append(int(gidx))

            if current_window:
                event_windows.append((current_window, last_event))

            # Frame from which cycle 1 disappears from event memory: the START of the
            # second window of the trigger event. Anchoring on the label start (not on
            # "the second window is already in history") means the keyframes collapse to
            # [tap1, tap2, grab] the instant the second scoop begins — the same context
            # the model had right before the first scoop.
            hide_from: int | None = None
            if self._skip_first_event_ids:
                starts = sorted(
                    w[0] for w, lbl in event_windows if lbl == self._skip_first_event_ids[0]
                )
                if len(starts) >= 2:
                    hide_from = starts[1]

            # For each frame, record the most recent ≤ event_memory_size candidate windows
            # from events that occurred strictly before this frame.
            for gidx in ep_global:
                before = [(w, l) for w, l in event_windows if w[0] < int(gidx)]
                if hide_from is not None and int(gidx) >= hide_from:
                    before = self._drop_first_occurrences(before)
                self._event_frame_candidates[int(gidx)] = [w for w, _ in before][-event_memory_size:]

        # Determine zero-pad shape from first sample.
        first = base_dataset[0]
        first_arr = np.asarray(first[image_keys[0]])
        # With delta_timestamps=[0.0] the shape is (1, C, H, W); without it, (C, H, W).
        self._single_frame_shape: tuple[int, ...] = (
            first_arr.shape[1:] if first_arr.ndim == 4 else first_arr.shape
        )

    def _drop_first_occurrences(
        self, events_before: list[tuple[list[int], int]]
    ) -> list[tuple[list[int], int]]:
        """Drop the first occurrence of each id in ``skip_first_event_ids``.

        mem10 demos pour twice into cup 1 (scoop→A, scoop→A, scoop→B), but the task is
        run one scoop per cup at inference. Applied from the second scoop's label start
        (see the caller), this makes the keyframe history read:

            [tap1, tap2, grab]                     -> about to scoop  (cycles 1 and 2 alike)
            [tap1, tap2, grab, scoop]              -> pour into cup 1 (cycles 1 and 2 alike)
            [tap1, tap2, grab, scoop, pour, scoop] -> pour into cup 2 (cycle 3)

        i.e. the one-scoop-per-cup structure inference actually sees, with cycles 1 and 2
        mapping to the same history AND the same cup, so nothing is ambiguous.
        """
        pending = set(self._skip_first_event_ids)
        kept: list[tuple[list[int], int]] = []
        for window, event_id in events_before:
            if event_id in pending:
                pending.discard(event_id)
                continue
            kept.append((window, event_id))
        return kept

    @property
    def hf_dataset(self):
        return self._base_dataset.hf_dataset

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = dict(self._base_dataset[idx])
        event_candidates = self._event_frame_candidates.get(idx, [])
        n_events = len(event_candidates)
        n_pad = self._event_memory_size - n_events

        zero = np.zeros(self._single_frame_shape, dtype=np.float32)
        drop_events = self._dropout_prob > 0.0 and np.random.random() < self._dropout_prob

        # Independent of the video-frame dropout above: forces the model to sometimes see
        # keyframes without a matching text summary (and vice versa), so it doesn't become
        # reliant on always getting both memory modalities together.
        if "phase_history" in sample and self._text_dropout_prob > 0.0 and np.random.random() < self._text_dropout_prob:
            sample["phase_history"] = "History: none"

        for key in self._image_keys:
            cur_arr = np.asarray(sample[key])
            cur_frame = cur_arr[0] if cur_arr.ndim == 4 else cur_arr  # (C, H, W)

            frames: list[np.ndarray] = [zero] * n_pad
            if drop_events:
                frames.extend([zero] * n_events)
            else:
                for cands in event_candidates:
                    # Randomly sample one frame from the candidate window for this event,
                    # matching the variability of when the event detector fires at rollout.
                    # Clip to frames strictly before the current frame to avoid sampling
                    # future frames when idx falls within the window of the current event.
                    valid = [f for f in cands if f < idx]
                    ev_idx = valid[np.random.randint(len(valid))] if valid else cands[0]
                    ev_arr = np.asarray(self._base_dataset[ev_idx][key])
                    frames.append(ev_arr[0] if ev_arr.ndim == 4 else ev_arr)
            frames.append(cur_frame)

            sample[key] = np.stack(frames, axis=0)  # (T, C, H, W)

        return sample


class TextDropoutDataset(torch.utils.data.Dataset):
    """Replaces ``phase_history`` with ``"History: none"`` with probability ``prob``.

    Same behaviour as the text dropout built into ``EventMemoryDataset``, but for configs
    that never build event-memory video stacks (``video_encoder=False``, so there is no
    ``EventMemoryDataset`` to fold it into). ``prob=1.0`` pins every sample to an empty
    history, which is how the no-textual-memory ablations keep the training prompt SHAPE
    identical to the memory arms while carrying no memory content: a missing
    ``phase_history`` key makes the tokenizer emit a different token sequence
    (``Task: …, State: …``) rather than an empty history, which would confound the
    ablation with a prompt-format change.
    """

    def __init__(self, dataset, prob: float) -> None:
        self._dataset = dataset
        self._prob = prob

    @property
    def hf_dataset(self):
        return self._dataset.hf_dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index) -> dict:
        sample = dict(self._dataset[int(index)])
        if "phase_history" in sample and np.random.random() < self._prob:
            sample["phase_history"] = "History: none"
        return sample


class ActionPaddingWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, num_padding=50, action_horizon=50):
        self._dataset = dataset
        self._num_padding = num_padding
        self._action_horizon = action_horizon

        logging.info(f"Padding dataset with {num_padding} 'stop' frames per episode.")

        if hasattr(dataset, "hf_dataset"):
            hf_ds = dataset.hf_dataset
        elif hasattr(dataset, "_dataset") and hasattr(dataset._dataset, "hf_dataset"):
            hf_ds = dataset._dataset.hf_dataset
        else:
            raise AttributeError("Dataset is missing the underlying hf_dataset.")

        ep_ids_np = np.array(hf_ds["episode_index"])
        unique_episodes = np.unique(ep_ids_np)

        self.new_indices = []
        self.padding_indices = set()
        # For each position in new_indices: number of real frames remaining in the
        # episode starting from (and including) this frame. 0 for padding frames.
        self.frames_to_end: list[int] = []

        current_global_idx = 0
        for ep_id in unique_episodes:
            ep_frames = np.where(ep_ids_np == ep_id)[0].tolist()
            n = len(ep_frames)

            for j, idx in enumerate(ep_frames):
                self.new_indices.append(int(idx))
                self.frames_to_end.append(n - j)  # remaining frames incl. current
                current_global_idx += 1

            last_real_idx = int(ep_frames[-1])
            for _ in range(self._num_padding):
                self.padding_indices.add(current_global_idx)
                self.new_indices.append(last_real_idx)
                self.frames_to_end.append(0)
                current_global_idx += 1

    def __len__(self):
        return len(self.new_indices)

    def __getitem__(self, index):
        idx = int(index)
        original_idx = self.new_indices[idx]
        sample = self._dataset[original_idx]

        if "actions" in sample:
            remaining = self.frames_to_end[idx]
            if remaining == 0 or remaining < self._action_horizon:
                # Padding frame (remaining==0) or real frame whose chunk extends past
                # episode end (remaining < H): zero out the tail that LeRobot clamped.
                actions = np.array(sample["actions"], dtype=np.float32)
                actions[remaining:] = 0.0
                sample["actions"] = actions

        return sample


def add_stop_padding(ds, num_frames=50, action_horizon=50):
    """Wraps a LeRobotDataset to append zero-action frames at the end of each episode
    and zero out the clamped tail of action chunks for real frames near episode end."""
    return ActionPaddingWrapper(ds, num_padding=num_frames, action_horizon=action_horizon)


def _rekey_episode_data_index(ds: lerobot_dataset.LeRobotDataset, episodes: list[int]) -> None:
    """Re-key ``ds.episode_data_index`` from positional to original-episode-index order.

    Works around a LeRobot bug: ``get_episode_data_index`` builds the from/to table
    *positionally* over the ``episodes`` subset (length == len(episodes)), but
    ``LeRobotDataset.__getitem__`` looks it up with the **original** ``episode_index``
    stored in the data. Those agree only when ``episodes`` is a contiguous prefix, so a
    sparse subset (e.g. dropping episodes 0, 2, 3) raises IndexError as soon as
    ``delta_timestamps`` are in play (which they always are here: action chunks and/or
    video history).

    The from/to offsets are already relative to the loaded subset, so they carry over
    unchanged — only the indexing changes. Slots for excluded episodes are unreachable
    (no frames from them exist in the dataset) and are left at zero.

    Must run *after* ``__init__`` — LeRobot's init-time ``check_timestamps_sync`` walks
    the table positionally and requires the original compact form.
    """
    src = ds.episode_data_index
    width = max(episodes) + 1
    from_ = torch.zeros(width, dtype=torch.long)
    to_ = torch.zeros(width, dtype=torch.long)
    for pos, ep in enumerate(episodes):
        from_[ep] = src["from"][pos]
        to_[ep] = src["to"][pos]
    ds.episode_data_index = {"from": from_, "to": to_}


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training.

    When ``data_config.repo_ids`` is provided, builds one ``LeRobotDataset`` per repo
    (each with its own per-dataset prompt-from-task transform) and concatenates them.
    Otherwise falls back to the single-repo path using ``data_config.repo_id``.
    """
    if data_config.repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    repo_ids: list[str] = []
    if data_config.repo_ids:
        repo_ids = list(data_config.repo_ids)
    elif data_config.repo_id is not None:
        repo_ids = [data_config.repo_id]
    if not repo_ids:
        raise ValueError("Repo ID is not set. Cannot create dataset.")

    sub_datasets: list[Dataset] = []
    for rid in repo_ids:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(rid)

        # Build delta_timestamps: action chunks + optional video history frames.
        delta_timestamps: dict[str, list[float]] = {
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        }
        if data_config.event_memory_training and data_config.video_image_keys:
            # Event-memory path: load a single frame per image key (offset 0.0)
            # so EventMemoryDataset can assemble the stack from event indices.
            for key in data_config.video_image_keys:
                delta_timestamps[key] = [0.0]
        elif data_config.video_image_keys and data_config.video_frame_offsets:
            for key in data_config.video_image_keys:
                delta_timestamps[key] = list(data_config.video_frame_offsets)

        # Non-destructive episode exclusion: load only the kept episodes for this repo.
        episodes = None
        if data_config.exclude_episodes and rid in data_config.exclude_episodes:
            excluded = set(data_config.exclude_episodes[rid])
            episodes = [i for i in range(dataset_meta.total_episodes) if i not in excluded]
            logging.info(
                "Excluding %d episode(s) from %s: %s (%d/%d kept)",
                len(excluded), rid, sorted(excluded), len(episodes), dataset_meta.total_episodes,
            )

        ds = lerobot_dataset.LeRobotDataset(rid, delta_timestamps=delta_timestamps, episodes=episodes)
        if episodes is not None:
            _rekey_episode_data_index(ds, episodes)

        if data_config.event_memory_training and data_config.video_image_keys:
            ds = EventMemoryDataset(
                ds,
                image_keys=data_config.video_image_keys,
                event_memory_size=data_config.event_memory_size,
                dropout_prob=data_config.event_dropout_prob,
                text_dropout_prob=data_config.text_dropout_prob,
                event_frame_window=data_config.event_frame_window,
                skip_first_event_ids=data_config.skip_first_event_ids,
            )
        elif data_config.text_dropout_prob > 0.0:
            # No event-memory wrapper to fold the text dropout into (the no-keyframe
            # ablations run with video_encoder=False), so apply it on its own.
            ds = TextDropoutDataset(ds, data_config.text_dropout_prob)

        # Append zero-action frames and zero-clamp tails near episode end (optional).
        if data_config.stop_padding:
            ds = add_stop_padding(ds, num_frames=data_config.stop_padding_frames, action_horizon=action_horizon)

        # Corrupt `state` in the same post-event window used for upsampling, so the model
        # can't shortcut a decision via proprioceptive drift instead of the memory signal.
        if (
            data_config.noise_state_in_upsample_window
            and data_config.upsample_after_event_id is not None
            and data_config.upsample_window_steps > 0
        ):
            state_stats = (data_config.norm_stats or {}).get("state")
            state_std = np.asarray(state_stats.std) if state_stats is not None else None
            # The noise window may be shorter than the upsample window: noise belongs
            # where the proprioceptive shortcut lives, but must end before the final
            # approach, or the arm loses the position accuracy it needs to hit the cup.
            noise_steps = data_config.noise_window_steps or data_config.upsample_window_steps
            ds = StateNoiseDataset(
                ds,
                data_config.upsample_after_event_id,
                noise_steps,
                state_std=state_std,
                window_offset=data_config.upsample_window_offset,
            )

        if data_config.prompt_from_task:
            ds = TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

        sub_datasets.append(ds)

    if len(sub_datasets) == 1:
        return sub_datasets[0]
    return typing.cast(Dataset, torch.utils.data.ConcatDataset(sub_datasets))


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions, _model.Labels | None]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def _repo_component_weights(
    concat_dataset: torch.utils.data.ConcatDataset,
    repo_weights: Sequence[float] | None,
) -> np.ndarray:
    """Per-frame weight array that balances draws across sub-datasets.

    Each frame in repo ``i`` receives weight ``repo_weights[i] / len(sub_dataset_i)``
    so that every repo contributes proportionally to ``repo_weights`` regardless of size.
    When ``repo_weights`` is ``None``, all repos are weighted equally.
    """
    sub_lens = [len(d) for d in concat_dataset.datasets]
    n = len(sub_lens)
    if repo_weights is None:
        repo_weights = [1.0] * n
    elif len(repo_weights) != n:
        raise ValueError(
            f"repo_weights has length {len(repo_weights)} but there are {n} sub-datasets."
        )
    if any(w < 0 for w in repo_weights):
        raise ValueError(f"repo_weights must be non-negative, got {repo_weights!r}")
    if sum(repo_weights) <= 0:
        raise ValueError(f"repo_weights must have a positive sum, got {repo_weights!r}")

    total = float(sum(repo_weights))
    per_sample = np.empty(sum(sub_lens), dtype=np.float64)
    offset = 0
    for w, l in zip(repo_weights, sub_lens, strict=True):
        if l > 0:
            per_sample[offset : offset + l] = (w / total) / l
        offset += l
    return per_sample


def _dataset_labels_and_episodes(ds) -> tuple[np.ndarray, np.ndarray] | None:
    """Best-effort ``(labels, episode_index)`` arrays aligned to ``ds``'s index space.

    Handles the full wrapper chain built by ``create_torch_dataset``: ``LeRobotDataset`` ->
    (optional) ``EventMemoryDataset`` -> (optional) ``ActionPaddingWrapper`` -> (optional)
    ``StateNoiseDataset`` -> (optional) ``TransformedDataset`` (from ``prompt_from_task``).
    Rather than special-casing each wrapper class, this unwraps generically via any
    ``._dataset`` attribute, remapping indices only for wrappers (like
    ``ActionPaddingWrapper``) that expose a ``new_indices`` array; everything else is
    assumed to preserve the index space 1:1. Returns ``None`` if no layer exposes
    ``hf_dataset`` (e.g. non-event datasets), so callers can skip event-window features
    gracefully.
    """
    hf_ds = getattr(ds, "hf_dataset", None)
    if hf_ds is not None:
        episodes = np.array(hf_ds["episode_index"])
        labels_raw = hf_ds["labels"]
        labels = np.array(
            [int(l.item()) if hasattr(l, "item") else (int(l[0]) if hasattr(l, "__len__") else int(l)) for l in labels_raw],
            dtype=np.int32,
        )
        return labels, episodes

    inner = getattr(ds, "_dataset", None)
    if inner is None:
        return None
    inner_info = _dataset_labels_and_episodes(inner)
    if inner_info is None:
        return None
    labels, episodes = inner_info
    new_indices = getattr(ds, "new_indices", None)
    if new_indices is not None:
        idx = np.asarray(new_indices, dtype=np.int64)
        return labels[idx], episodes[idx]
    return labels, episodes


def _event_window_mask_for_dataset(
    ds, event_id: int, window_steps: int, window_offset: int = 0
) -> np.ndarray | None:
    """Boolean mask, ``True`` for the ``window_steps`` frames starting ``window_offset``
    frames after EACH labeled run of ``event_id`` in an episode (``False`` elsewhere).

    ``window_offset`` skips the stretch right after the event where the decision has not
    been made yet. For mem10 the arm is still leaving the bowl for ~45 frames after the
    scoop run ends, and the target is not yet recoverable from the trajectory there
    (between/within cup spread stays below 1.0), so boosting those frames spends weight
    on samples that teach nothing about cup selection.

    ``labels`` marks a short window around each event transition (see
    ``examples/libero/label_dataset_libero.py``), so "the end of a labeled run of ``event_id``" is a good
    proxy for "right after that event finished" — e.g. event_id=3 ("retracted") anchors
    the frames where the arm has to commit to moving left vs. right.

    An event id may repeat within an episode (mem10 labels all three scoops id 3), so every
    run gets its own window rather than only the last one; anchoring solely on the final
    run would miss the first scoop, which is where the cup decision is actually made. For
    an event that occurs exactly once this is identical to the previous behaviour.
    """
    info = _dataset_labels_and_episodes(ds)
    if info is None:
        return None
    labels, episodes = info
    mask = np.zeros(len(labels), dtype=bool)
    for ep in np.unique(episodes):
        ep_idx = np.where(episodes == ep)[0]  # ascending -> chronological order within the episode
        event_positions = np.where(labels[ep_idx] == event_id)[0]
        if event_positions.size == 0:
            continue
        # End of each contiguous run of event_id.
        run_ends = [
            int(p)
            for i, p in enumerate(event_positions)
            if i + 1 == len(event_positions) or event_positions[i + 1] != p + 1
        ]
        for anchor in run_ends:
            # window_offset may be NEGATIVE, to reach frames before the event run ends —
            # for mem10 the most memory-dependent stretch is the first ~12% of the
            # scoop->pour transport, which lies before the anchor. Clamp to [0, len):
            # a negative slice start would silently wrap and mask the END of the episode
            # instead, boosting exactly the wrong frames with no error.
            start = max(0, anchor + 1 + window_offset)
            end = min(len(ep_idx), start + window_steps)
            if end > start:
                mask[ep_idx[start:end]] = True
    return mask


class StateNoiseDataset(torch.utils.data.Dataset):
    """Wraps a dataset and adds Gaussian noise to ``state`` for frames inside the
    post-``event_id`` window (see ``_event_window_mask_for_dataset``).

    ``state`` (EE position/orientation/gripper) can leak which way a decision should go
    even after the arm has physically returned to a neutral pose (e.g. it retracts to a
    slightly different spot depending on which side it just grabbed/dropped from). That
    gives the policy a proprioceptive shortcut around the intended vision/phase_history
    memory signal. Corrupting ``state`` in that window forces it to rely on the memory
    signal instead for whatever decision falls inside it.
    """

    # Fraction of each state dimension's own dataset std (from norm_stats) to use as that
    # dimension's noise std. A single flat noise magnitude is wrong here: e.g. for LIBERO's
    # 8-dim state (ee_xyz, axis-angle rot_xyz, 2x gripper_qpos), gripper std is ~0.007 while
    # ee position std is ~0.10-0.13 — a flat noise big enough to matter for position would be
    # 4-5x the gripper channel's entire natural range and wipe out its signal. Scaling by each
    # dimension's own std keeps the perturbation proportionate. Falls back to a flat default
    # when norm_stats aren't available (e.g. a config that hasn't computed them yet).
    STATE_NOISE_FRAC = 0.5
    _FALLBACK_NOISE_STD = 0.03

    def __init__(
        self,
        dataset,
        event_id: int,
        window_steps: int,
        state_std: np.ndarray | None = None,
        window_offset: int = 0,
    ):
        mask = _event_window_mask_for_dataset(dataset, event_id, window_steps, window_offset)
        if mask is None:
            raise ValueError(
                "noise_state_in_upsample_window requires the dataset to have "
                "`labels`/`episode_index` columns."
            )
        self._dataset = dataset
        self._mask = mask
        if state_std is not None:
            self._noise_std = self.STATE_NOISE_FRAC * np.asarray(state_std, dtype=np.float32)
        else:
            logging.warning(
                "StateNoiseDataset: no norm_stats for `state` available; falling back to a "
                "flat noise std of %.3f for every state dimension.", self._FALLBACK_NOISE_STD,
            )
            self._noise_std = None

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        idx = int(index)
        sample = dict(self._dataset[idx])
        if self._mask[idx] and "state" in sample:
            state = np.asarray(sample["state"], dtype=np.float32)
            noise_std = self._noise_std if self._noise_std is not None else self._FALLBACK_NOISE_STD
            noise = (np.random.standard_normal(state.shape) * noise_std).astype(np.float32)
            sample["state"] = state + noise
        return sample


def _event_window_weights_for_dataset(
    ds, event_id: int, window_steps: int, boost_weight: float, window_offset: int = 0
) -> np.ndarray | None:
    """Weight-1 everywhere, except ``boost_weight`` inside the post-``event_id`` window
    (see ``_event_window_mask_for_dataset``)."""
    mask = _event_window_mask_for_dataset(ds, event_id, window_steps, window_offset)
    if mask is None:
        return None
    weights = np.ones(len(mask), dtype=np.float64)
    weights[mask] = boost_weight
    return weights


def _compute_event_window_weights(
    dataset, event_id: int, window_steps: int, boost_weight: float, window_offset: int = 0
) -> np.ndarray | None:
    """Like ``_event_window_weights_for_dataset``, but also handles multi-repo ``ConcatDataset``."""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        parts = [
            _event_window_weights_for_dataset(d, event_id, window_steps, boost_weight, window_offset)
            for d in dataset.datasets
        ]
        if any(p is None for p in parts):
            return None
        return np.concatenate(parts)
    return _event_window_weights_for_dataset(dataset, event_id, window_steps, boost_weight, window_offset)


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions, _model.Labels | None]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.

    Yields batches ``(observation, actions, labels)``. ``labels`` is ``int32`` per batch element when
    the dataset provides ``labels``; otherwise ``None``. For event training, ``labels < 0`` (e.g. ``-1``)
    maps to an extra logits class and is supervised; see ``Pi0.compute_loss_event``.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    raw_dataset = dataset
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    # Multi-repo weighted sampling. Only applies when explicit repo_weights are set.
    # When repo_weights is None, sampling is uniform over all frames (proportional to
    # demo count * demo length), which is the natural behaviour for merged datasets.
    sampler_weights: np.ndarray | None = None
    if (
        sampler is None
        and data_config.repo_weights is not None
        and data_config.repo_ids
        and isinstance(raw_dataset, torch.utils.data.ConcatDataset)
    ):
        sampler_weights = _repo_component_weights(raw_dataset, data_config.repo_weights)
        sub_lens = [len(d) for d in raw_dataset.datasets]
        logging.info(
            "Multi-repo weighted sampling enabled. repos=%s sub_lens=%s weights=%s",
            list(data_config.repo_ids),
            sub_lens,
            list(data_config.repo_weights),
        )

    # Event-window upsampling: boost the sampling weight of frames shortly after a chosen
    # event (e.g. the decision window right after "retracted"), so the model sees
    # that moment more often than its natural share of the episode would give it.
    if (
        sampler is None
        and data_config.upsample_after_event_id is not None
        and data_config.upsample_window_steps > 0
    ):
        event_weights = _compute_event_window_weights(
            raw_dataset,
            data_config.upsample_after_event_id,
            data_config.upsample_window_steps,
            data_config.upsample_weight,
            data_config.upsample_window_offset,
        )
        if event_weights is None:
            logging.warning(
                "upsample_after_event_id=%d is set but the dataset has no `labels`/"
                "`episode_index` columns; skipping event-window upsampling.",
                data_config.upsample_after_event_id,
            )
        else:
            if sampler_weights is None:
                sampler_weights = np.ones(len(event_weights), dtype=np.float64)
            sampler_weights = sampler_weights * event_weights
            n_boosted = int(np.sum(event_weights > 1.0))
            logging.info(
                "Event-window upsampling enabled: event_id=%d window_steps=%d weight=%.2f boosted_frames=%d/%d",
                data_config.upsample_after_event_id,
                data_config.upsample_window_steps,
                data_config.upsample_weight,
                n_boosted,
                len(event_weights),
            )

    if sampler is None and sampler_weights is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sampler_weights),
            num_samples=int(sampler_weights.size),
            replacement=True,
            generator=generator,
        )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions, _model.Labels | None]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"], batch.get("labels")
