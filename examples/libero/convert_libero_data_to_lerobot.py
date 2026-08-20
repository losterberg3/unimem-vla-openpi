"""Convert LIBERO-Mem HDF5 demos (downloaded from the source HF dataset repo) to
one LeRobot dataset per task file.

Usage:
    uv run examples/libero/convert_libero_data_to_lerobot.py

For converting raw LIBERO (non-Mem) HDF5 demonstrations you already have on disk,
use convert_libero_hdf5_to_lerobot.py instead — this script fetches its source
files from the Hugging Face dataset named by SOURCE_REPO_ID.
"""

import fnmatch
import os
from pathlib import Path
import shutil
import h5py
import numpy as np
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download, list_repo_files
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro

REPO_OWNER = "lars"
SOURCE_REPO_ID = "libero-mem/LIBERO-Mem"

def _resize_flip_uint8(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] != (256, 256):
        image = np.asarray(PILImage.fromarray(image).resize((256, 256)))
    # Rotate 180 degrees to match official LIBERO orientation
    #image = image[::-1, ::-1]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image

def _build_state(demo: h5py.Group, step_idx: int) -> np.ndarray | None:
    # LIBERO-Mem specific proprioception keys
    joints = None
    for k in ["obs/joint_states", "joint_states", "obs/robot0_joint_pos"]:
        if k in demo:
            joints = np.asarray(demo[k][step_idx])
            break
            
    gripper = None
    for k in ["obs/gripper_states", "gripper_states", "obs/robot0_gripper_qpos"]:
        if k in demo:
            gripper = np.asarray(demo[k][step_idx])
            break

    if joints is None:
        return None
    
    if gripper is None:
        gripper = np.array([0.0, 0.0], dtype=np.float32)
    if gripper.ndim == 0:
        gripper = np.array([gripper, gripper], dtype=np.float32)

    # Combine: 7 (joints) + 2 (gripper) = 9 total
    state = np.concatenate([joints, gripper]).astype(np.float32)
    
    # Force to 8D to match LeRobot dataset definition
    if state.shape[0] > 8:
        state = state[:8]
    elif state.shape[0] < 8:
        state = np.pad(state, (0, 8 - state.shape[0]))
    return state

def main(
    source_repo_id: str = SOURCE_REPO_ID,
    *,
    repo_owner: str = REPO_OWNER,
    file_pattern: str = "*_demo.hdf5",
    max_files: int | None = None, # Set to 1 for your local disk safety
    downsample: bool = True,
    push_to_hub: bool = False,
):
    all_files = list_repo_files(repo_id=source_repo_id, repo_type="dataset")
    hdf5_files = [f for f in all_files if fnmatch.fnmatch(Path(f).name, file_pattern)]
    hdf5_files.sort()
    if max_files is not None:
        hdf5_files = hdf5_files[:max_files]

    print(f"Found {len(hdf5_files)} file(s) to convert.")
    step_stride = 2 if downsample else 1
    
    for remote_file in hdf5_files:
        task_name = Path(remote_file).stem.replace("_demo", "").replace("_", " ")
        output_repo_id = f"{repo_owner}/{Path(remote_file).stem.lower()}"
        
        # Create dataset
        dataset = LeRobotDataset.create(
            repo_id=output_repo_id,
            robot_type="panda",
            fps=10 if downsample else 20,
            features={
                "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            }
        )

        local_file = hf_hub_download(repo_id=source_repo_id, repo_type="dataset", filename=remote_file)
        
        with h5py.File(local_file, "r") as h5f:
            demo_keys = sorted(k for k in h5f["data"].keys() if k.startswith("demo_"))
            
            for demo_key in demo_keys:
                demo = h5f[f"data/{demo_key}"]
                num_steps = demo["actions"].shape[0]
                
                # Language from metadata
                task_inst = demo.attrs.get("language_instruction", task_name)

                for step_idx in range(0, num_steps, step_stride):
                    # Image: Check LIBERO-Mem 'rgb' keys first
                    img = None
                    for k in ["obs/agentview_rgb", "obs/agentview_image", "agentview_rgb"]:
                        if k in demo:
                            img = _resize_flip_uint8(np.array(demo[k][step_idx]))
                            break
                    
                    # Wrist: Check LIBERO-Mem 'rgb' keys first
                    w_img = None
                    for k in ["obs/eye_in_hand_rgb", "obs/eye_in_hand_image", "eye_in_hand_rgb"]:
                        if k in demo:
                            w_img = _resize_flip_uint8(np.array(demo[k][step_idx]))
                            break
                    
                    if img is None: break
                    if w_img is None: w_img = img.copy()

                    state = _build_state(demo, step_idx)
                    
                    # Action handling with downsampling sum
                    action = np.array(demo["actions"][step_idx]).astype(np.float32)
                    if downsample and (step_idx + 1) < num_steps:
                        next_act = np.array(demo["actions"][step_idx+1]).astype(np.float32)
                        action[:6] += next_act[:6] # Sum deltas
                        action[6:] = next_act[6:]  # Latest gripper
                    
                    if action.shape[0] > 7: action = action[:7]
                    elif action.shape[0] < 7: action = np.pad(action, (0, 7 - action.shape[0]))

                    dataset.add_frame({
                        "image": img,
                        "wrist_image": w_img,
                        "state": state,
                        "actions": action,
                        "task": task_inst,
                    })

                dataset.save_episode()
        print(f"Finished {output_repo_id}")

if __name__ == "__main__":
    tyro.cli(main)