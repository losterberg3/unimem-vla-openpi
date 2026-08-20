"""Teleoperated xArm demo recording -> LeRobot dataset.

Runs as a plain top-level script (not wrapped in main()): import-time side effects
parse hardware CLI flags, then connect to the arm and cameras immediately, then the
collection loop runs until Ctrl+C. Start/stop/tap are driven by touching flag files
in /tmp (see START_FLAG/STOP_FLAG/TAP_FLAG below) so an external teleop process can
control recording without a shared Python process.

Built and tested on one specific physical rig (one UFactory xArm + two Intel
RealSense cameras). Hardware identity (arm IP, camera serials) is overridable via
CLI flags (--arm-ip / --camera-serial-external / --camera-serial-wrist); dataset
repo name and task description are still plain constants in the Config block below
since they change per recording session, not per rig. See examples/xarm/README.md
for adapting to different hardware.

Usage:
    uv run examples/xarm/demo_collection.py
    uv run examples/xarm/demo_collection.py --arm-ip 192.168.1.220 \\
        --camera-serial-external <serial> --camera-serial-wrist <serial>
"""

import dataclasses
import time
import sys
import select
import numpy as np
import pyrealsense2 as rs
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from xarm.wrapper import XArmAPI
from pathlib import Path

# ------------------------
# Hardware identity — override via CLI, not by editing these defaults. Find
# RealSense serials with `rs-enumerate-devices | grep Serial`.
# ------------------------

@dataclasses.dataclass
class HardwareConfig:
    arm_ip: str = "192.168.1.219"
    camera_serial_external: str = "244222071219"
    camera_serial_wrist: str = "025222070771"


_hw = tyro.cli(HardwareConfig)
ARM_IP = _hw.arm_ip
SERIAL_EXTERNAL = _hw.camera_serial_external
SERIAL_WRIST = _hw.camera_serial_wrist

# ------------------------
# Task config — EDIT PER RECORDING SESSION (dataset repo name, task description).
# ------------------------

REPO_NAME = "<your-hf-username>/mem10"
FPS = 20.0
DT = 1.0 / FPS

#TASK_DESCRIPTION = "measure the length of the hammer using the tape measurer and control the retract to ensure the hook doesn't slam"
#TASK_DESCRIPTION = "put three scoops of beans into the mixing bowl"
#TASK_DESCRIPTION = "take the yellow bottle off the table, and then wipe where it was"
TASK_DESCRIPTION = "watch the human tap a cup, then put a scoop of beans in that cup"

START_FLAG = Path("/tmp/start_demo")
STOP_FLAG  = Path("/tmp/stop_demo")

# Human-event marker (e.g. tapping a cup): press Enter in this terminal or
# touch /tmp/cup_tap while recording to flag the current frame.
TAP_FLAG = Path("/tmp/cup_tap")

if START_FLAG.exists():
    START_FLAG.unlink()

if STOP_FLAG.exists():
    STOP_FLAG.unlink()

if TAP_FLAG.exists():
    TAP_FLAG.unlink()


# ------------------------
# Init robot + cameras
# ------------------------
arm = XArmAPI(ARM_IP)
arm.connect()

# Connect to cameras
camera_pipelines = {}

# Enable streams only for our specific serials
for serial in [SERIAL_EXTERNAL, SERIAL_WRIST]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
    
    try:
        pipeline.start(config)
        camera_pipelines[serial] = pipeline
        print(f"Started camera: {serial}")
    except Exception as e:
        print(f"Failed to start camera {serial}: {e}")
        sys.exit(1)

def read_cameras():
    # Fetch frames by serial number to ensure no swapping
    frames_exterior = camera_pipelines[SERIAL_EXTERNAL].wait_for_frames()
    frames_wrist = camera_pipelines[SERIAL_WRIST].wait_for_frames()
    wrist = frames_wrist.get_color_frame()
    exterior = frames_exterior.get_color_frame()
    wrist = np.asanyarray(wrist.get_data())
    exterior = np.asanyarray(exterior.get_data())
    exterior2 = np.zeros_like(exterior)
    return wrist, exterior, exterior2

def timed_input(prompt, timeout, default="y"):
    print(prompt, end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip().lower()
    else:
        print(f"\nNo response after {timeout}s → defaulting to '{default}'")
        return default

def poll_tap():
    """Non-blocking check for a human-event marker (Enter key or /tmp/cup_tap)."""
    tapped = False
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        tapped = True
    if TAP_FLAG.exists():
        TAP_FLAG.unlink()
        tapped = True
    return tapped

# ------------------------
# Create dataset
# ------------------------

dataset_path = HF_LEROBOT_HOME / REPO_NAME

if dataset_path.exists(): 
    dataset = LeRobotDataset(
        root=dataset_path,
        repo_id=REPO_NAME,
    )
    print("Adding to existing dataset, waiting for signal.")
else:
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xarm",
        fps=FPS,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": { # this one is not used, put it as zeros or something
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # We will use joint *velocity* actions here (6D)
                "names": ["actions"],
            },
            "human_event": {  # 1.0 on frames where the human marked an event (cup tap)
                "dtype": "float32",
                "shape": (1,),
                "names": ["human_event"],
            },
        },
    )
    print("Dataset created, waiting for start signal.")

# Enable async image writer. Required so clear_episode_buffer() actually
# deletes orphan PNGs on a discard (LeRobot gates that cleanup on
# image_writer is not None).
dataset.start_image_writer(num_processes=0, num_threads=4)

# ------------------------
# Collect episode
# ------------------------

recording = False
prev_data = None  # Buffer to hold the observation for time t

try:
    while True:
        if START_FLAG.exists() and not recording:
            START_FLAG.unlink()
            print("Starting demo")
            recording = True
            prev_data = None # Reset buffer for new demo
            poll_tap()  # drain stale keypresses / tap flag from before the demo

        if STOP_FLAG.exists() and recording:
            STOP_FLAG.unlink()
            print("Ending demo")
            
            recording = False
            prev_data = None # Clear buffer
            resp = timed_input("Save this demo? [y/n]: ", timeout=6, default="y")

            if resp == "y":
                dataset.save_episode()
                print("Episode saved")
            else:
                dataset.clear_episode_buffer()
                print("Episode discarded")

        if not recording:
            time.sleep(0.05)
            continue

        start = time.perf_counter()

        # 1. Capture CURRENT state (Time t+1 relative to prev_data)
        #joints = arm.get_servo_angle(is_radian=True)[1][:6]
        pose = arm.get_position()[1]
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360
        # ensure roll and yaw are continuous, also make sure pitch doesn't exceed 90 deg
        # when collecting demos
        angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()

        curr_state = np.array(pose[:3] + angles_rad, dtype=np.float32)
        
        wrist, base, base2 = read_cameras()

        tapped = poll_tap()
        if tapped:
            print("Human event marked (cup tap)")

        gripper = (arm.get_gripper_position()[1] - 850) / -860
        gripper_np = np.array([gripper], dtype=np.float32)
        curr = np.concatenate([curr_state, gripper_np]).astype(np.float32)

        # 2. If we have a previous observation, record it with CURRENT state as the action
        if prev_data is not None:
            dataset.add_frame(
                {
                    "state": prev_data["state"],
                    "gripper_position": prev_data["gripper"],
                    "actions": curr,  # This is the "future" state reached
                    "exterior_image_1_left": prev_data["base"],
                    "exterior_image_2_left": prev_data["base2"],
                    "wrist_image_left": prev_data["wrist"],
                    "human_event": prev_data["tap"],
                    "task": TASK_DESCRIPTION,
                }
            )

        # 3. Store current observations to be paired with the next frame's state
        prev_data = {
            "state": curr[:6],
            "gripper": curr[-1:],
            "wrist": wrist,
            "base": base,
            "base2": base2,
            "tap": np.array([float(tapped)], dtype=np.float32),
        }

        # ---- Timing ----
        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, DT - elapsed))

except KeyboardInterrupt:
    print("Shutting down")

finally:
    for p in camera_pipelines.values():
        p.stop()
    arm.disconnect()