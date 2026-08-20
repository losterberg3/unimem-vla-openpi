"""
Convert raw LIBERO HDF5 demonstrations to LeRobot format.

Usage:
uv run examples/libero/convert_libero_hdf5_to_lerobot.py --data_dir demonstration_data --repo_id your_username/lego_demos

If the LeRobot dataset directory already exists, new episodes are **appended** by default (no delete).
Columns present in the dataset but not produced from HDF5 (e.g. ``labels``, ``phase_history``, ``ee_pos`` after
``label_dataset`` / ``lerobot_meta_add_features``) are filled with neutral defaults for new frames only.

Use ``--overwrite`` to remove an existing dataset and recreate from scratch (previous behavior).
"""

import shutil
from pathlib import Path

import datasets as hf_datasets
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES
from PIL import Image as PILImage

# Base keys always produced from LIBERO HDF5 in this script.
_CORE_FEATURE_KEYS = frozenset({"image", "wrist_image", "state", "actions", "task"})


def _feature_shape_tuple(spec: dict) -> tuple[int, ...]:
    sh = spec.get("shape", [])
    return tuple(int(x) for x in sh)


def _default_value_for_feature(spec: dict):
    """Scalar / vector / image placeholder matching LeRobot ``features`` metadata."""
    dtype = spec["dtype"]
    shape = _feature_shape_tuple(spec)
    if dtype == "float32":
        return np.zeros(shape, dtype=np.float32)
    if dtype == "int64":
        return np.zeros(shape, dtype=np.int64)
    if dtype == "string":
        return ""
    if dtype == "image":
        # Channel-last uint8, same convention as the rest of this script.
        if len(shape) == 3 and shape[2] == 3:
            h, w, _ = shape
            return np.zeros((h, w, 3), dtype=np.uint8)
        if len(shape) == 3 and shape[0] == 3:
            c, h, w = shape
            return np.zeros((h, w, c), dtype=np.uint8)
        return np.zeros(shape, dtype=np.uint8)
    if dtype == "video":
        raise ValueError(
            "Appending to a dataset with dtype='video' camera columns is not supported from this HDF5 "
            "converter; use an image-stored dataset or recreate with --overwrite."
        )
    raise ValueError(f"Unsupported dtype for default fill: {dtype!r}")


def _build_frame_row(dataset: LeRobotDataset, core: dict) -> dict:
    """All non-DEFAULT_FEATURES keys plus ``task`` must appear in each frame dict for ``add_frame``."""
    feats = dataset.features
    required = (set(feats) - set(DEFAULT_FEATURES.keys())) | {"task"}
    out: dict = {}
    for name in required:
        if name == "task":
            out["task"] = core["task"]
            continue
        if name in core:
            out[name] = core[name]
        else:
            out[name] = _default_value_for_feature(feats[name])
    return out


def _validate_append_compatible(dataset: LeRobotDataset, *, expect_fps: int) -> None:
    if int(dataset.fps) != int(expect_fps):
        raise ValueError(
            f"Existing dataset fps={dataset.fps} does not match converter output fps={expect_fps}. "
            "Recreate with matching --downsample / dataset, or use --overwrite."
        )
    feats = dataset.features
    missing = [k for k in _CORE_FEATURE_KEYS if k != "task" and k not in feats]
    if missing:
        raise ValueError(f"Existing dataset is missing required keys {missing}; cannot append from HDF5.")

    def _expect_image(name: str, exp_hw: tuple[int, int, int]) -> None:
        spec = feats[name]
        if spec.get("dtype") != "image":
            raise ValueError(f"Existing feature {name!r} must be dtype 'image' for this converter, got {spec!r}.")
        sh = _feature_shape_tuple(spec)
        if sh != exp_hw and sh != (exp_hw[2], exp_hw[0], exp_hw[1]):
            # Allow CHW vs HWC for strict check: only HWC (256,256,3) expected here.
            if sh != exp_hw:
                raise ValueError(f"Existing {name} shape {sh} != expected {exp_hw} (HWC).")

    _expect_image("image", (256, 256, 3))
    _expect_image("wrist_image", (256, 256, 3))

    st = feats["state"]
    if st.get("dtype") != "float32" or _feature_shape_tuple(st) != (8,):
        raise ValueError(f"Existing 'state' must be float32 (8,), got {st}.")

    ac = feats["actions"]
    if ac.get("dtype") != "float32" or _feature_shape_tuple(ac) != (7,):
        raise ValueError(f"Existing 'actions' must be float32 (7,), got {ac}.")


def main(
    data_dir: str,
    repo_id: str = "your_username/libero_lego_demos",
    *,
    push_to_hub: bool = False,
    fix_freeplay_mismatch: bool = False,
    downsample: bool = True,
    overwrite: bool = False,
):
    """Convert LIBERO HDF5 demonstrations to LeRobot format.

    Args:
        data_dir: Directory containing robosuite demonstration folders with demo.hdf5 files
        repo_id: Repository ID for the LeRobot dataset
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        fix_freeplay_mismatch: If True, handle data where free-play actions were recorded but
            observations weren't (uses last N actions to match observation count)
        downsample: If True, take every other frame (20 Hz -> 10 Hz) to match standard LIBERO
        overwrite: If True, delete an existing dataset at ``HF_LEROBOT_HOME/repo_id`` before converting.
            If False and the path exists, **append** new episodes; extra dataset-only columns get defaults.
    """
    data_path = Path(data_dir)

    # Find all demo.hdf5 files (check current dir and one level down)
    demo_files = list(data_path.glob("*/demo.hdf5"))
    if (data_path / "demo.hdf5").exists():
        demo_files.append(data_path / "demo.hdf5")
    print(f"Found {len(demo_files)} demonstration files")

    if len(demo_files) == 0:
        print(f"No demo.hdf5 files found in {data_dir}")
        return

    output_path = HF_LEROBOT_HOME / repo_id
    expect_fps = 10 if downsample else 20

    if output_path.exists() and overwrite:
        print(f"Removing existing dataset at {output_path} (--overwrite)")
        shutil.rmtree(output_path)

    if output_path.exists() and not overwrite:
        print(f"Appending to existing dataset at {output_path}")
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=str(output_path),
            download_videos=False,
        )
        _validate_append_compatible(dataset, expect_fps=expect_fps)
        # When loaded from existing parquets, image columns in hf_dataset have the raw
        # Arrow struct schema {bytes, path} instead of datasets.Image(). Cast them so
        # that hf_features returns Image() objects — path strings then encode correctly
        # in from_dict and concatenate_datasets sees matching schemas on both sides.
        for _img_key in ("image", "wrist_image"):
            if _img_key in dataset.hf_dataset.features:
                dataset.hf_dataset = dataset.hf_dataset.cast_column(_img_key, hf_datasets.Image())
        dataset.start_image_writer(num_processes=5, num_threads=10)
    else:
        # Create LeRobot dataset (new tree)
        print(f"Creating new dataset at {output_path}")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=expect_fps,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (8,),  # 7 DOF + gripper
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),  # 6 DOF + gripper
                    "names": ["action"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    # Process each demonstration file
    episode_idx = 0
    for demo_file in sorted(demo_files):
        folder_name = demo_file.parent.name
        print(f"\nProcessing {folder_name}")

        # Extract task description from folder name
        parts = folder_name.split("_")
        task_start_idx = None
        for i, part in enumerate(parts):
            if part in ("pick", "place", "put", "move", "open", "close", "push", "pull", "lift", "stack"):
                task_start_idx = i
                break

        if task_start_idx:
            task_description = " ".join(parts[task_start_idx:])
        else:
            task_description = "pick and place lego"
        print(f"  Task: {task_description}")

        with h5py.File(demo_file, "r") as f:
            demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]
            print(f"  Found {len(demo_keys)} episodes in file")

            for demo_key in demo_keys:
                demo = f[f"data/{demo_key}"]

                if "obs" not in demo:
                    print(f"    Skipping {demo_key}: no observations")
                    continue

                if "language_instruction" in demo.attrs:
                    task_description = demo.attrs["language_instruction"]
                    print(f"    Using language from HDF5: {task_description}")

                if not task_description or task_description.strip() == "":
                    print(f"    Skipping {demo_key}: empty task description")
                    continue

                num_action_steps = demo["actions"].shape[0]
                action_offset = 0

                if fix_freeplay_mismatch:
                    obs_len = None
                    for img_key in ["obs/agentview_image", "obs/frontview_image", "obs/image"]:
                        if img_key in demo:
                            obs_len = demo[img_key].shape[0]
                            break

                    if obs_len and obs_len < num_action_steps:
                        action_offset = num_action_steps - obs_len
                        print(
                            f"    WARNING: actions ({num_action_steps}) > obs ({obs_len}), using last {obs_len} steps"
                        )
                        num_steps = obs_len
                    else:
                        num_steps = num_action_steps
                else:
                    num_steps = num_action_steps

                step_stride = 2 if downsample else 1
                frame_indices = list(range(0, num_steps, step_stride))
                actual_frames = len(frame_indices)

                if downsample:
                    print(
                        f"    {demo_key}: {num_steps} steps -> {actual_frames} frames (downsampled), task: {task_description}"
                    )
                else:
                    print(f"    {demo_key}: {num_steps} steps, task: {task_description}")

                for step_idx in frame_indices:
                    image = None
                    for img_key in ["obs/agentview_image", "obs/frontview_image", "obs/image"]:
                        if img_key in demo:
                            image = np.array(demo[img_key][step_idx])
                            break

                    if image is None:
                        image_keys = [k for k in demo["obs"].keys() if "image" in k.lower() and "eye" not in k.lower()]
                        if image_keys:
                            image = np.array(demo[f"obs/{image_keys[0]}"][step_idx])
                        else:
                            raise ValueError(f"No base image observation found in {demo_key}")

                    if image.shape[:2] != (256, 256):
                        image = np.array(PILImage.fromarray(image).resize((256, 256)))

                    image = image[::-1, ::-1]

                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)

                    wrist_image = None
                    for wrist_key in ["obs/robot0_eye_in_hand_image", "obs/eye_in_hand_image", "obs/wrist_image"]:
                        if wrist_key in demo:
                            wrist_image = np.array(demo[wrist_key][step_idx])
                            break

                    if wrist_image is None:
                        wrist_keys = [k for k in demo["obs"].keys() if "eye" in k.lower() or "wrist" in k.lower()]
                        if wrist_keys:
                            wrist_image = np.array(demo[f"obs/{wrist_keys[0]}"][step_idx])
                        else:
                            print("    Warning: No wrist camera found, using agentview as fallback")
                            wrist_image = image.copy()

                    if wrist_image.shape[:2] != (256, 256):
                        wrist_image = np.array(PILImage.fromarray(wrist_image).resize((256, 256)))

                    wrist_image = wrist_image[::-1, ::-1]

                    if wrist_image.dtype != np.uint8:
                        wrist_image = wrist_image.astype(np.uint8)

                    eef_pos = None
                    for pos_key in ["obs/robot0_eef_pos", "obs/eef_pos"]:
                        if pos_key in demo:
                            eef_pos = np.array(demo[pos_key][step_idx])
                            break

                    eef_quat = None
                    for quat_key in ["obs/robot0_eef_quat", "obs/eef_quat"]:
                        if quat_key in demo:
                            eef_quat = np.array(demo[quat_key][step_idx])
                            break

                    if eef_pos is not None and eef_quat is not None:
                        q = eef_quat.copy()
                        if q[3] > 1.0:
                            q[3] = 1.0
                        elif q[3] < -1.0:
                            q[3] = -1.0
                        den = np.sqrt(1.0 - q[3] * q[3])
                        if den < 1e-8:
                            axis_angle = np.zeros(3)
                        else:
                            axis_angle = (q[:3] * 2.0 * np.arccos(q[3])) / den
                        robot_state = np.concatenate([eef_pos, axis_angle])
                    else:
                        joint_pos = None
                        for state_key in ["obs/robot0_joint_pos", "obs/joint_pos"]:
                            if state_key in demo:
                                joint_pos = np.array(demo[state_key][step_idx])
                                break
                        if joint_pos is None:
                            raise ValueError(f"No EEF or joint position found in {demo_key}")
                        robot_state = joint_pos

                    gripper = None
                    for gripper_key in ["obs/robot0_gripper_qpos", "obs/gripper_qpos"]:
                        if gripper_key in demo:
                            gripper = np.array(demo[gripper_key][step_idx])
                            break

                    if gripper is None:
                        gripper = np.array([0.0])

                    state = np.concatenate([robot_state, gripper])
                    if state.shape[0] > 8:
                        state = state[:8]
                    elif state.shape[0] < 8:
                        state = np.pad(state, (0, 8 - state.shape[0]))

                    action = np.array(demo["actions"][step_idx + action_offset])

                    if downsample and (step_idx + 1 + action_offset) < len(demo["actions"]):
                        next_action = np.array(demo["actions"][step_idx + 1 + action_offset])
                        action[:6] = action[:6] + next_action[:6]
                        action[6:] = next_action[6:]

                    if action.shape[0] > 7:
                        action = action[:7]
                    elif action.shape[0] < 7:
                        action = np.pad(action, (0, 7 - action.shape[0]))

                    core = {
                        "image": image,
                        "wrist_image": wrist_image,
                        "state": state.astype(np.float32),
                        "actions": action.astype(np.float32),
                        "task": task_description,
                    }
                    dataset.add_frame(_build_frame_row(dataset, core))

                dataset.save_episode()
                episode_idx += 1

    if getattr(dataset, "image_writer", None) is not None:
        dataset.stop_image_writer()

    print(f"\n{'=' * 60}")
    print(f"✓ Wrote {episode_idx} episodes to LeRobot format")
    print(f"✓ Dataset root: {output_path}")
    print(f"{'=' * 60}")

    if push_to_hub:
        print(f"\nPushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("✓ Dataset pushed to Hub")


if __name__ == "__main__":
    tyro.cli(main)
