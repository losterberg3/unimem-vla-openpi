#!/usr/bin/env python3
"""xArm event-conditioned inference against a served policy checkpoint.

Event tracking matches libero_inference.py. Built and tested on one specific
physical rig (one UFactory xArm + two Intel RealSense cameras) — the hardware
identity fields on ``Args`` (arm_ip, camera_serial_external, camera_serial_wrist,
home_x/y/z/roll/pitch/yaw, collision_sensitivity) are what our particular rig
happened to be; override them from the CLI or an experiment yaml for your own
setup instead of editing the defaults below. See examples/xarm/README.md for
adapting this to a different arm or camera.
"""

import collections
import dataclasses
import json
import logging
import re
from typing import Literal
import pathlib
import sys
import threading
import time

import imageio.v2 as imageio
import numpy as np
import pyrealsense2 as rs
import tyro
from openpi_client import websocket_client_policy
from xarm.wrapper import XArmAPI

# ------------------------
# Hardware / timing config
# ------------------------
FPS = 20.0
DT = 1.0 / FPS
CONTROL_HZ = 40.0
PREDICTION_HORIZON = 20     # action horizon used in guided_inference blending
MIN_EXECUTION_HORIZON = 2  # steps to execute before triggering a replan
_DELAY_INIT = 5             # initial delay estimate fed into the latency queue
_THREAD_BUFFER_SIZE = 5     # rolling window size for tracking inference latency

# Physical-rig defaults: our arm's IP, our two RealSense cameras' serials, and our
# arm's calibrated home pose. These are almost certainly wrong for your setup — they
# exist as Args defaults (arm_ip, camera_serial_external, camera_serial_wrist,
# home_x/y/z/roll/pitch/yaw) purely so the script has something to run with out of
# the box. Override them via CLI flags or an experiment yaml; don't edit them here.
# Find RealSense serials with `rs-enumerate-devices | grep Serial`.
ARM_IP = "192.168.1.219"
SERIAL_EXTERNAL = "244222071219"
SERIAL_WRIST = "025222070771"

HOME_POSITION = dict(x=465.4, y=0.0, z=388.7, roll=-178.1, pitch=0.0, yaw=-179.9)

# Gripper snap-to-close: if the policy outputs a normalized close command above this
# threshold, snap it up to GRIPPER_CLOSE_SNAP so the gripper actually grips hard.
GRIPPER_CLOSE_THRESHOLD = 0.4
GRIPPER_CLOSE_SNAP = 0.7

# xArm collision sensitivity: 0=off, 1=very low, 2=low, 3=moderate, 4=high, 5=very high.
# Must be re-applied after every mode switch (the arm resets it internally).
COLLISION_SENSITIVITY = 1

# ------------------------
# Event config
# ------------------------
# Per-task completion-language vocabularies.  Add a new dict here whenever a
# new dataset is labeled; the key must match the ``task`` field in Args / yaml.
COMPLETION_LANGUAGE_MEM7: dict[int, str] = {
    0: "grabbed tape measure",
    1: "attached to hammer",
    2: "extended tape",
    3: "retracted tape",
    4: "placed tape measure",
}

COMPLETION_LANGUAGE_MEM8: dict[int, str] = {
    0: "picked up spoon",
    1: "scooped beans",
    2: "poured beans",
    3: "placed spoon",
}

COMPLETION_LANGUAGE_MEM9: dict[int, str] = {
    0: "grabbed bottle",
    1: "grabbed sponge",
    2: "wiped table",
    3: "placed sponge",
}

# Must stay in sync with MEM10V4_COMPLETION_LANGUAGE in scripts/label_dataset_xarm.py.
# One tap, one grab, one scoop, one pour, one place — every event occurs exactly
# once, so nothing has to be counted or disambiguated.
COMPLETION_LANGUAGE_MEM10: dict[int, str] = {
    0: "human tap",
    1: "grabbed spoon",
    2: "scooped beans",
    3: "poured beans",
    4: "placed spoon",
}

TASK_COMPLETION_LANGUAGE: dict[str, dict[int, str]] = {
    "mem7": COMPLETION_LANGUAGE_MEM7,
    "mem8": COMPLETION_LANGUAGE_MEM8,
    "mem9": COMPLETION_LANGUAGE_MEM9,
    "mem10": COMPLETION_LANGUAGE_MEM10,
}

# Keep the old name as an alias so nothing else in this file needs to change
# for the default (mem7) path.
COMPLETION_LANGUAGE = COMPLETION_LANGUAGE_MEM7

_EVENT_CONFIDENCE_THRESHOLD = 0.8

# ------------------------
# Globals populated in main()
# ------------------------
policy: websocket_client_policy.WebsocketClientPolicy | None = None
arm: XArmAPI | None = None
camera_pipelines: dict[str, rs.pipeline] = {}
_run_prompt: str = ""


# ------------------------
# Event helpers (mirrors libero_inference.py)
# ------------------------

def _predicted_event_from_probs(
    probs: np.ndarray,
    num_event_classes: int,
    threshold: float = _EVENT_CONFIDENCE_THRESHOLD,
) -> int:
    event_probs = np.asarray(probs, dtype=np.float64).reshape(-1)[:num_event_classes]
    k = int(np.argmax(event_probs))
    ignore_id = num_event_classes  # one past the last valid class
    return k if float(event_probs[k]) > threshold else ignore_id


def _format_event_history(completed_chunks: list[str]) -> str:
    history = ", ".join(completed_chunks)
    return f"History: {history}" if history else "History: none"


def guided_inference(
    observation: dict,
    action_prev: np.ndarray,
    delay: int,
    time_since_last_inference: int,
) -> tuple[np.ndarray, dict]:
    """Blend the stale action chunk with fresh policy output.

    W transitions from 1 (pure old actions) for the first ``delay`` steps,
    then decays with an exponential curve through the blend region, then drops
    to 0 for steps that have already been executed. The result smoothly hands
    off from previously-committed actions to the new policy plan.

    Returns (blended_actions, infer_output) where blended_actions has shape
    (PREDICTION_HORIZON, dof).
    """
    assert policy is not None
    H = PREDICTION_HORIZON
    dof = action_prev.shape[1] if action_prev.ndim == 2 else 7

    blend_end = H - time_since_last_inference
    i = np.arange(delay, blend_end)
    W = np.ones(H)
    W[blend_end:] = 0.0
    if len(i) > 0:
        c = (blend_end - i) / (blend_end - delay + 1)
        W[delay:blend_end] = c * (np.exp(c) - 1) / (np.exp(1) - 1)

    if action_prev.shape[0] < H:
        action_prev = np.pad(action_prev, ((0, H - action_prev.shape[0]), (0, 0)), mode="constant")

    infer_output = policy.infer(observation)
    v_pi = np.array(infer_output["actions"], dtype=np.float32)
    if v_pi.ndim == 3:
        v_pi = v_pi[0]
    v_pi = v_pi[:H, :dof]

    action_estimate = action_prev[:H, :dof] * W[:, None] + v_pi * (1 - W[:, None])
    return action_estimate, infer_output


# ------------------------
# Home
# ------------------------

def _go_home() -> None:
    """Move arm to home position and open gripper, then restore servo mode for inference."""
    assert arm is not None
    logging.info("Going home")
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    arm.set_gripper_enable(enable=True)
    arm.set_gripper_mode(0)
    arm.set_position(**HOME_POSITION, speed=100, is_radian=False, wait=True)
    arm.set_gripper_position(850.0, wait=True)
    arm.set_mode(1)
    arm.set_state(0)
    arm.set_collision_sensitivity(COLLISION_SENSITIVITY)  # re-apply after mode switch; mode changes reset this
    logging.info("Home reached")


# ------------------------
# Camera
# ------------------------

def _start_cameras() -> None:
    global camera_pipelines
    camera_pipelines = {}
    for serial in [SERIAL_EXTERNAL, SERIAL_WRIST]:
        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        rs_cfg.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
        try:
            pipeline.start(rs_cfg)
            camera_pipelines[serial] = pipeline
            logging.info("Started camera: %s", serial)
        except Exception as e:
            logging.error("Failed to start camera %s: %s", serial, e)
            sys.exit(1)


# ------------------------
# Observation
# ------------------------

def get_observation(event_history: str | None = None) -> dict:
    assert arm is not None and camera_pipelines

    frames_exterior = camera_pipelines[SERIAL_EXTERNAL].wait_for_frames()
    frames_wrist = camera_pipelines[SERIAL_WRIST].wait_for_frames()
    exterior = np.asanyarray(frames_exterior.get_color_frame().get_data())
    wrist = np.asanyarray(frames_wrist.get_color_frame().get_data())

    pose = arm.get_position()[1]
    pose[3] = pose[3] % 360
    pose[5] = pose[5] % 360
    angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
    state = np.array(pose[:3] + angles_rad, dtype=np.float32)

    gripper_pos = np.array([(arm.get_gripper_position()[1] - 850) / -860], dtype=np.float32)

    obs = {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/state": state,
        "observation/gripper_position": gripper_pos,
        "prompt": _run_prompt,
    }
    if event_history is not None:
        obs["phase_history"] = event_history
    return obs


# ------------------------
# Action execution
# ------------------------

def interpolate_action(state: np.ndarray, goal: np.ndarray) -> None:
    """Smooth servo-cartesian interpolation from state to goal (both in degrees for RPY)."""
    delta_increment = (goal - state) / (DT * CONTROL_HZ)

    for _ in range(int(DT * CONTROL_HZ)):
        start = time.perf_counter()
        command = state + delta_increment
        command[3] = (command[3] + 180) % 360 - 180
        command[5] = (command[5] + 180) % 360 - 180

        x, y, z, roll, pitch, yaw = command

        arm.set_servo_cartesian(command, speed=100, mvacc=1000)
        #print(command)
        time_left = (1 / CONTROL_HZ) - (time.perf_counter() - start)
        time.sleep(max(time_left, 0))


# ------------------------
# Args
# ------------------------

@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None
    task: str = "mem7"   # selects event vocabulary; must be a key in TASK_COMPLETION_LANGUAGE
    prompt: str = "measure the length of the hammer using the tape measurer and control the retract to ensure the hook doesn't slam"
    num_rollouts: int = 5
    max_episode_steps: int = 1500
    event_confidence_threshold: float = _EVENT_CONFIDENCE_THRESHOLD
    # Log the raw event-head output every inference, including events that get
    # discarded by the confidence threshold or the repeat-suppression guard (both
    # of which are otherwise silent). Diagnostic only.
    debug_event_probs: bool = False
    # Set True when the checkpoint was trained with video_encoder=True (expects T frames).
    video_encoder: bool = False
    num_frames: int = 4
    # Inference mode — controls what memory is sent to the model each step:
    #   "no_memory"    — no event history text, no historical frames (prompt + current frame only).
    #   "text"         — event history text only; zeros for video history slots.
    #   "text_keyframe"   — event history text + keyframes always; requires video_encoder=True.
    #   "keyframe"        — keyframes always, NO event history text; requires video_encoder=True.
    #   "video" — rolling frames every frame_stride_sec seconds, NO text;
    #                    requires video_encoder=True.
    mode: Literal["no_memory", "text", "text_keyframe", "keyframe", "video"] = "text"
    # Seconds between consecutive video-encoder frames. MUST equal the training
    # data config's `frame_stride_sec` (see LeRobotXarmVideoDataConfig): training
    # declares the stride in SECONDS and LeRobot resolves it to frame offsets via
    # the dataset fps, so seconds — not steps — is the quantity that must match.
    frame_stride_sec: float | None = None
    # Deprecated: stride in execution steps. Only used as a fallback when
    # frame_stride_sec is unset, via stride_steps/FPS. This drifts whenever the
    # execution loop misses its FPS deadline, which is why it is not the default.
    stride_steps: int = 20
    # NOTE: history slots that have not been filled yet are sent as ZEROED images in
    # raw uint8 space, not as a repeat of the opening frame. The server normalizes with
    # uint8/255*2-1 (models/model.py), so a zero slot arrives at exactly -1.
    # Save one PNG of the external camera each time a new event is appended to the
    # history, into <out_path>/<date>/<time>/rollout_XXX/ alongside a metadata.json.
    save_keyframes: bool = True
    out_path: str = "rollout_keyframes"
    gripper_close_threshold: float = GRIPPER_CLOSE_THRESHOLD
    gripper_close_snap: float = GRIPPER_CLOSE_SNAP
    # --- Hardware identity: override these for your own arm/cameras (see module
    # docstring). Defaults below are just this rig's values.
    arm_ip: str = ARM_IP
    camera_serial_external: str = SERIAL_EXTERNAL
    camera_serial_wrist: str = SERIAL_WRIST
    home_x: float = HOME_POSITION["x"]
    home_y: float = HOME_POSITION["y"]
    home_z: float = HOME_POSITION["z"]
    home_roll: float = HOME_POSITION["roll"]
    home_pitch: float = HOME_POSITION["pitch"]
    home_yaw: float = HOME_POSITION["yaw"]
    collision_sensitivity: int = COLLISION_SENSITIVITY  # 0=off .. 5=very high


# ------------------------
# Keyframe saving
# ------------------------

def _make_run_out_dir(base: str) -> pathlib.Path:
    """Per-run output directory (<base>/<date>/<time>) so runs never overwrite each other."""
    now = time.localtime()
    return pathlib.Path(base) / time.strftime("%Y-%m-%d", now) / time.strftime("%H-%M-%S", now)


def _slugify(label: str) -> str:
    """'first scoop' -> 'first_scoop' (filesystem-safe, lowercase)."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "event"


def _keyframe_image_name(order: int, label: str, occurrence: int, view: str, t_sec: float) -> str:
    """Ordered, self-describing filename: 03_first_scoop_1_t12.3s_exterior.png.

    ``order`` is the 1-based position in the event history (so the files sort in
    the order the events actually fired); ``occurrence`` counts how many times
    this same label has fired so far in the rollout; ``t_sec`` is the elapsed
    time since the rollout started, in seconds; ``view`` is the camera
    ("exterior" or "wrist").
    """
    return f"{order:02d}_{_slugify(label)}_{occurrence}_t{t_sec:07.1f}s_{view}.png"


def _save_keyframe_image(out_dir: pathlib.Path, name: str, frame_rgb: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out_dir / name), np.ascontiguousarray(frame_rgb))
    logging.info("  Saved event image: %s", out_dir / name)


# ------------------------
# Event memory
# ------------------------

def _build_keyframe_obs(
    obs: dict,
    keyframe_frames_ext: list[np.ndarray],
    keyframe_frames_wrist: list[np.ndarray],
    total_frames: int,
) -> dict:
    """Replace scalar image keys with (T, H, W, C) stacks for the video encoder.

    Layout: [zero-padded slots] + [event frames] + [current frame].
    RoPE in TemporalAttentionBlock encodes frame order, so the most recent frame
    is always at the end regardless of how many events have fired.
    """
    n_keyframes = len(keyframe_frames_ext)
    n_pad = total_frames - 1 - n_keyframes   # empty slots before first event

    cur_ext = obs["observation/exterior_image_1_left"]
    cur_wrist = obs["observation/wrist_image_left"]
    h, w, c = cur_ext.shape

    pad_ext = [np.zeros((h, w, c), dtype=np.uint8)] * n_pad
    pad_wrist = [np.zeros((h, w, c), dtype=np.uint8)] * n_pad

    obs = dict(obs)
    obs["observation/exterior_image_1_left"] = np.stack(
        pad_ext + list(keyframe_frames_ext) + [cur_ext], axis=0,  # (T, H, W, C)
    )
    obs["observation/wrist_image_left"] = np.stack(
        pad_wrist + list(keyframe_frames_wrist) + [cur_wrist], axis=0,
    )
    return obs


def _resolve_frame_stride_sec(args: Args) -> float:
    """Seconds between consecutive video-encoder frames, matching training.

    Training declares the stride in seconds (``frame_stride_sec`` on the data
    config); LeRobot turns it into ``delta_timestamps`` offsets of
    ``-(num_frames-1-i) * stride`` and resolves those against the dataset fps.
    Counting the stride in execution steps instead only agrees while the hardware
    loop holds exactly FPS — and it does not, since each step blocks on two
    RealSense ``wait_for_frames`` calls plus arm round-trips. Seconds is the
    quantity that has to match, so it is what we sample on.
    """
    if args.frame_stride_sec is not None:
        return float(args.frame_stride_sec)
    fallback = args.stride_steps / FPS
    logging.warning(
        "video mode: frame_stride_sec is unset — falling back to stride_steps/FPS "
        "= %d/%.0f = %.2fs. This assumes the execution loop holds exactly %.0f Hz. "
        "Set frame_stride_sec to the training config's value instead.",
        args.stride_steps, FPS, fallback, FPS,
    )
    return fallback


# ------------------------
# Setup / rollout / inference
# ------------------------

def _setup(args: Args) -> None:
    """Connect to policy server, arm, and cameras. Called once per experiment."""
    global policy, arm, _run_prompt, SERIAL_EXTERNAL, SERIAL_WRIST, HOME_POSITION, COLLISION_SENSITIVITY
    _run_prompt = args.prompt
    # Hardware identity comes from Args (CLI/yaml), not the module-level defaults —
    # _start_cameras/get_observation/_go_home read the globals below, so those need
    # to be updated before anything touches the cameras or the arm's home pose.
    SERIAL_EXTERNAL = args.camera_serial_external
    SERIAL_WRIST = args.camera_serial_wrist
    HOME_POSITION = dict(
        x=args.home_x, y=args.home_y, z=args.home_z,
        roll=args.home_roll, pitch=args.home_pitch, yaw=args.home_yaw,
    )
    COLLISION_SENSITIVITY = args.collision_sensitivity
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logging.info("Connected to policy server; metadata: %s", policy.get_server_metadata())
    arm = XArmAPI(args.arm_ip)
    arm.connect()
    if arm.get_state() != 0:
        arm.clean_error()
        time.sleep(0.5)
    arm.motion_enable(enable=True)
    arm.set_collision_sensitivity(COLLISION_SENSITIVITY)
    arm.set_mode(1)
    arm.set_state(0)
    _start_cameras()


def run_rollout(args: Args, rollout_idx: int = 0, out_dir: pathlib.Path | None = None) -> dict:
    """Run one rollout using a threaded inference + execution loop.

    A background inference thread replans every MIN_EXECUTION_HORIZON steps
    using guided_inference to blend the stale action chunk with the fresh
    policy output. The execution thread runs at FPS, drives the arm, and
    signals the inference thread via a condition variable.

    Press Enter to end the rollout early; Ctrl+C also works.
    """
    _vocab = TASK_COMPLETION_LANGUAGE.get(args.task, COMPLETION_LANGUAGE_MEM7)
    _num_event_classes = len(_vocab)
    # _event_tracking: run event head, detect events, append them to the internal
    #   history, and save keyframe images. no_memory joins this set so it logs the
    #   same events/images as the memory modes — it just never feeds them back.
    # _feed_text_history: send a phase_history string to the policy at all. The model
    #   was trained with phase_history ALWAYS present (LeRobotXarmEventKeyframeDataConfig
    #   maps it in; the start of every episode reads "History: none"). Omitting the key
    #   changes the tokenized prompt ("Task: …, State: …" vs "Task: …, History: none,
    #   State: …"), which is off-distribution from step 0 — so every event-tracking mode
    #   must feed it, even the no-memory baseline.
    # _dynamic_text_history: let that string reflect the real completed events. When
    #   False (keyframe / no_memory) it stays pinned to "History: none" for the whole
    #   rollout — no textual memory reaches the policy, but the prompt shape matches.
    _event_tracking = args.mode in ("text", "text_keyframe", "keyframe", "no_memory")
    _feed_text_history = args.mode in ("text", "text_keyframe", "keyframe", "no_memory")
    _dynamic_text_history = args.mode in ("text", "text_keyframe")

    _go_home()
    input(f"\nRollout {rollout_idx + 1}/{args.num_rollouts}: set up scene then press Enter to start... ")

    done_event = threading.Event()

    # --- manual gripper override: while set, the execution thread forces the
    # gripper open and ignores the policy's gripper command. Toggled from stdin. ---
    _gripper_open_override = threading.Event()

    def _stdin_loop() -> None:
        """Read operator commands during the rollout.

        Empty line (bare Enter) ends the rollout; 'g' opens the gripper and holds
        it open; 'c' hands the gripper back to the policy.
        """
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "":
                done_event.set()
                return
            if cmd in ("g", "o", "open"):
                _gripper_open_override.set()
                print("  [gripper] MANUAL OPEN — policy gripper command ignored (type 'c' to release)")
            elif cmd in ("c", "close", "release", "resume"):
                _gripper_open_override.clear()
                print("  [gripper] override released — policy back in control")
            else:
                print("  [stdin] 'g' = open gripper, 'c' = release override, Enter = end rollout")
        done_event.set()

    threading.Thread(target=_stdin_loop, daemon=True).start()
    print("  [running — Enter ends this rollout | 'g'+Enter opens gripper | 'c'+Enter releases]")

    # --- shared action buffer and step counter (protected by _cv) ---
    _mutex = threading.Lock()
    _cv = threading.Condition(_mutex)
    _t: int = 0                                             # steps executed since last replan
    _observation_curr: dict = {}
    _action_curr = np.zeros((PREDICTION_HORIZON, 7), dtype=np.float32)

    # --- event state (written by inference thread under _event_lock) ---
    _event_lock = threading.Lock()
    _completed_event_chunks: list[str] = []
    _last_appended_event: int | None = None

    # --- mutable rollout stats (owned by execution thread except where noted) ---
    step = 0
    infer_latencies: list[float] = []
    exec_periods: list[float] = []   # realized wall-clock length of each execution step
    exec_busy: list[float] = []      # work time per step, excluding the sleep
    ended_by = "max_steps"
    rollout_start = time.perf_counter()

    # --- keyframe image capture (written by the inference thread under _event_lock) ---
    if out_dir is None:
        out_dir = _make_run_out_dir(args.out_path)
    _keyframe_img_dir = out_dir / f"rollout_{rollout_idx:03d}"
    _keyframe_records: list[dict] = []          # one entry per saved PNG
    _label_counts: dict[str, int] = {}       # label -> times fired so far

    # --- keyframe memory (written by inference thread on events, read each inference) ---
    _keyframe_frames_ext: list[np.ndarray] = []
    _keyframe_frames_wrist: list[np.ndarray] = []
    _keyframe_lock = threading.Lock()

    # --- video frame buffer (video mode) — fixed-grid, matching libero_inference.py ---
    # A frame is captured once every frame_stride_sec and then STAYS PUT in its slot
    # until it is pushed off the end by newer captures. The stored frames do not move
    # relative to each other and are never re-picked, so the history a given inference
    # sees only changes at capture boundaries — not continuously. Slots are seeded with
    # zeros, and the live frame is appended as the final slot at inference time.
    #
    # Deliberately NOT a sliding "last N seconds" window: that re-anchors every slot on
    # every replan, which drifts off-distribution. This is the LIBERO behaviour.
    _video_mode: bool = bool(args.video_encoder and args.mode == "video")
    _video_stride_sec = _resolve_frame_stride_sec(args) if _video_mode else 0.0
    _video_slots = max(args.num_frames - 1, 0)
    _video_buf_ext: collections.deque = collections.deque(maxlen=_video_slots)
    _video_buf_wrist: collections.deque = collections.deque(maxlen=_video_slots)
    _video_buf_age: collections.deque = collections.deque(maxlen=_video_slots)  # capture times
    _video_lock = threading.Lock()
    # Trigger the first capture immediately, so the grid starts at rollout t=0
    # (libero seeds _last_stride_step to -stride_steps for the same reason).
    _video_last_capture: float = -1e9
    _video_captures = 0

    def _init_video_buffer(observation: dict) -> None:
        """Zero-fill the history slots (libero pre-fills with np.zeros_like)."""
        ext = observation["observation/exterior_image_1_left"]
        wrist = observation["observation/wrist_image_left"]
        with _video_lock:
            _video_buf_ext.clear()
            _video_buf_wrist.clear()
            _video_buf_age.clear()
            for _ in range(_video_slots):
                _video_buf_ext.append(np.zeros_like(ext))
                _video_buf_wrist.append(np.zeros_like(wrist))
                _video_buf_age.append(None)

    def _capture_video_frame(now: float, observation: dict) -> None:
        """Append to the fixed grid once per stride (execution thread)."""
        nonlocal _video_last_capture, _video_captures
        if now - _video_last_capture < _video_stride_sec:
            return
        _video_last_capture = now
        _video_captures += 1
        with _video_lock:
            # maxlen evicts the oldest slot; surviving frames keep their relative order.
            _video_buf_ext.append(observation["observation/exterior_image_1_left"].copy())
            _video_buf_wrist.append(observation["observation/wrist_image_left"].copy())
            _video_buf_age.append(now)
        logging.info(
            "  [video] captured grid frame #%d at t=%.1fs", _video_captures, now - rollout_start,
        )

    # -----------------------------------------------------------------------
    # _get_action: called by execution thread each step.
    # Increments _t, stores the latest observation so the inference thread can
    # use it, wakes the inference thread, and returns the buffered action.
    # -----------------------------------------------------------------------
    def _get_action(observation_new: dict) -> np.ndarray:
        nonlocal _t, _observation_curr
        with _cv:
            _t += 1
            _observation_curr = observation_new
            _cv.notify()
            # Clamp to buffer bounds: the inference thread may lag behind
            # (e.g. a slow replan) while execution keeps ticking.
            return _action_curr[min(_t - 1, PREDICTION_HORIZON - 1), :].copy()

    # -----------------------------------------------------------------------
    # Inference thread: replans every MIN_EXECUTION_HORIZON steps via
    # guided_inference, then updates the shared action buffer.
    # Also owns event tracking.
    # -----------------------------------------------------------------------
    def _inference_loop() -> None:
        nonlocal _t, _action_curr, _completed_event_chunks, _last_appended_event

        # Set when a new event fires; tells the policy server to slide its
        # stateful hidden-state cache on the NEXT inference call (and only that call).
        _pending_keyframe_slide: bool = False
        # Event-image metadata held over until that next call. The event is only known
        # after the call that produced it, so the flag — and therefore the frame the
        # server actually appends to its cache — belongs to the FOLLOWING request.
        # Saving that frame is what makes the saved keyframes show the policy's real memory
        # rather than the frame the detector happened to fire on.
        _pending_keyframe_save: dict | None = None

        def _save_keyframe_snapshot(meta: dict, ext_img, wrist_img, t_captured: float) -> None:
            """Write one event snapshot. Filenames keep the DETECTION time so ordering
            still lines up with the history log; the record carries the captured frame's
            own timestamp and the lag between them."""
            ext_name = _keyframe_image_name(meta["order"], meta["chunk"], meta["occurrence"], "exterior", meta["t_sec"])
            wrist_name = _keyframe_image_name(meta["order"], meta["chunk"], meta["occurrence"], "wrist", meta["t_sec"])
            _save_keyframe_image(_keyframe_img_dir, ext_name, ext_img)
            _save_keyframe_image(_keyframe_img_dir, wrist_name, wrist_img)
            _keyframe_records.append({
                "index": meta["order"],
                "exterior_file": ext_name,
                "wrist_file": wrist_name,
                "event_label": meta["chunk"],
                "event_id": meta["event_id"],
                "occurrence": meta["occurrence"],
                "step": meta["step"],
                "t_sec": round(meta["t_sec"], 2),
                # Timestamp of the image actually written, and how far it trails detection.
                "t_sec_image": round(t_captured, 2),
                "image_lag_s": round(t_captured - meta["t_sec"], 2),
                "image_is_cached_keyframe": meta["deferred"],
            })

        Q: collections.deque = collections.deque([_DELAY_INIT], maxlen=_THREAD_BUFFER_SIZE)

        # Ages (negative seconds) of the frames actually sent, for logging. None marks a
        # still-zero-filled slot. The newest gap is 0..stride, not exactly stride —
        # that is inherent to a fixed grid and matches libero.
        _video_realized_offsets: list[float | None] = []

        while not done_event.is_set():
            with _cv:
                while _t < MIN_EXECUTION_HORIZON and not done_event.is_set():
                    _cv.wait(timeout=0.1)
                if done_event.is_set():
                    return
                time_since_last = _t
                action_prev = _action_curr[time_since_last:PREDICTION_HORIZON, :].copy()
                delay = max(Q)
                obs = {**_observation_curr}   # shallow copy; arrays not mutated before inference

            # Inject event history text into the observation copy. Dynamic modes
            # (text / text_keyframe) send the real completed-event history; pinned
            # modes (keyframe / no_memory) always send "History: none" so the prompt
            # matches training without leaking any textual memory to the policy.
            if _feed_text_history:
                if _dynamic_text_history:
                    with _event_lock:
                        chunks = list(_completed_event_chunks)
                    obs["phase_history"] = _format_event_history(chunks)
                else:
                    obs["phase_history"] = _format_event_history([])

            # Stash the raw (single-frame) images before event memory injection so
            # we can save them if a new event fires after this inference call.
            raw_ext = obs["observation/exterior_image_1_left"]
            raw_wrist = obs["observation/wrist_image_left"]

            # Keyframe modes: the policy server holds a stateful hidden-state cache and only
            # needs the current frame each call. no_memory/text never have a real
            # event to slide on (_pending_keyframe_slide stays False forever for
            # them below), so their cache is seeded once with zero history and then
            # never touched again — same result as always-zero-padded full recompute,
            # but O(1) per step instead of reprocessing the whole window every call.
            if args.video_encoder and args.mode in ("text_keyframe", "keyframe", "no_memory", "text"):
                obs["new_keyframe"] = _pending_keyframe_slide
                if _pending_keyframe_slide and _pending_keyframe_save is not None:
                    # This request's frame is the one the server appends to its hidden-state cache,
                    # so THIS is the image that becomes the policy's keyframe memory.
                    _save_keyframe_snapshot(
                        _pending_keyframe_save, raw_ext, raw_wrist,
                        time.perf_counter() - rollout_start,
                    )
                    _pending_keyframe_save = None
                _pending_keyframe_slide = False
            elif _video_mode:
                # Read the fixed grid as-is and append the live frame. No re-selection:
                # the historical slots are byte-identical between captures, so successive
                # replans within one stride see exactly the same history.
                now = time.perf_counter()
                with _video_lock:
                    hist_ext = list(_video_buf_ext)
                    hist_wrist = list(_video_buf_wrist)
                    ages = list(_video_buf_age)

                # Ages of the frames actually sent, for logging. None = zero-filled slot.
                _video_realized_offsets = [
                    (None if a is None else a - now) for a in ages
                ] + [0.0]

                # Stack the (H, W, C) frames along a new temporal axis → (T, H, W, C),
                # matching _build_keyframe_obs. The server adds the batch dim, yielding the
                # (B, T, H, W, C) the video encoder expects. (Stacking on axis=1 would
                # fold the image height into the temporal dim and corrupt the batch.)
                obs["observation/exterior_image_1_left"] = np.stack(
                    hist_ext + [raw_ext], axis=0,
                )
                obs["observation/wrist_image_left"] = np.stack(
                    hist_wrist + [raw_wrist], axis=0,
                )
            # (no remaining video_encoder modes need client-side stacking: keyframe
            # modes and no_memory/text rely on the server's stateful cache above,
            # and "video" mode is handled by the stride-stacking branch.)

            t_infer = time.perf_counter()
            action_new, infer_output = guided_inference(obs, action_prev, delay, time_since_last)
            lat = time.perf_counter() - t_infer
            infer_latencies.append(lat)
            if _video_mode:
                logging.info(
                    "[infer] step~%d | infer=%.3fs | frame ages (s): %s",
                    step, lat,
                    ["zero" if o is None else round(o, 1) for o in _video_realized_offsets],
                )
            else:
                logging.info(
                    "[infer] step~%d | infer=%.3fs | %s",
                    step, lat,
                    obs.get("phase_history", "event tracking off"),
                )

            # Update shared buffer and reset step counter (outside the lock,
            # matching the threaded_inference.py pattern).
            _action_curr[: action_new.shape[0], :] = action_new
            _t -= time_since_last
            Q.append(_t)

            # Event tracking: detect new events, update event history text, and
            # (when mode uses keyframes) save the current visual frame as a snapshot.
            if _event_tracking and "event_id" in infer_output:
                predicted_event = _predicted_event_from_probs(
                    infer_output["event_id"],
                    num_event_classes=_num_event_classes,
                    threshold=args.event_confidence_threshold,
                )
                # DIAGNOSTIC: raw head output before thresholding/dedup, so an event that
                # is predicted but then discarded is still visible. Distinguishes
                # "below threshold" from "suppressed as a repeat of the last append"
                # (the dedup below is silent). Remove once the tap issue is settled.
                if args.debug_event_probs:
                    _probs = np.asarray(infer_output["event_id"], dtype=np.float64).reshape(-1)[
                        :_num_event_classes
                    ]
                    _rank = np.argsort(_probs)[::-1][:2]
                    _would_dedup = predicted_event == _last_appended_event
                    logging.info(
                        "  [event] argmax=%d(%s) p=%.3f | 2nd=%d(%s) p=%.3f | thr=%.2f "
                        "| passed=%s | last_appended=%s%s",
                        _rank[0], _vocab.get(int(_rank[0]), "?"), _probs[_rank[0]],
                        _rank[1], _vocab.get(int(_rank[1]), "?"), _probs[_rank[1]],
                        args.event_confidence_threshold,
                        predicted_event in _vocab,
                        _last_appended_event,
                        "  <-- SUPPRESSED as repeat" if _would_dedup else "",
                    )
                with _event_lock:
                    if (
                        predicted_event in _vocab
                        and predicted_event != _last_appended_event
                    ):
                        chunk = _vocab[predicted_event]
                        _completed_event_chunks.append(chunk)
                        _last_appended_event = predicted_event
                        logging.info(
                            "  Event → '%s' | history: %s", chunk,
                            _format_event_history(_completed_event_chunks),
                        )
                        # Snapshot the external camera at the moment this event was
                        # appended to the history. Naming keeps both firing order and
                        # per-label repetition (e.g. 03_first_scoop_1.png).
                        if args.save_keyframes:
                            occurrence = _label_counts.get(chunk, 0) + 1
                            _label_counts[chunk] = occurrence
                            _keyframe_mode = args.video_encoder and args.mode in (
                                "text_keyframe", "keyframe"
                            )
                            _meta = {
                                "order": len(_completed_event_chunks),
                                "chunk": chunk,
                                "occurrence": occurrence,
                                "event_id": int(predicted_event),
                                "step": step,
                                "t_sec": time.perf_counter() - rollout_start,
                                "deferred": _keyframe_mode,
                            }
                            if _keyframe_mode:
                                # Hold it: the frame the server caches arrives on the next
                                # call, together with new_keyframe=True.
                                _pending_keyframe_save = _meta
                            else:
                                # No keyframe cache in this mode — nothing is retained, so
                                # the detection frame is the only meaningful snapshot.
                                _save_keyframe_snapshot(_meta, raw_ext, raw_wrist, _meta["t_sec"])
                        # A real event fired: tell the server to slide its stateful
                        # hidden-state cache (evict oldest, append this frame) on the NEXT call.
                        # _keyframe_frames_ext is kept only for logging/video-overlay purposes
                        # now — the server's own cache is the source of truth for memory.
                        if args.video_encoder and args.mode in ("text_keyframe", "keyframe"):
                            _pending_keyframe_slide = True
                            with _keyframe_lock:
                                if len(_keyframe_frames_ext) < args.num_frames - 1:
                                    _keyframe_frames_ext.append(raw_ext.copy())
                                    _keyframe_frames_wrist.append(raw_wrist.copy())
                                    logging.info(
                                        "  [video] Saved frame #%d/%d for '%s'",
                                        len(_keyframe_frames_ext), args.num_frames - 1,
                                        _vocab[predicted_event],
                                    )

        # Rollout ended before the deferred save could ride along with a following call
        # (last event fired on the final inference). Fall back to that call's frame so
        # the final event still gets an image, and mark it as not-the-cached-keyframe.
        if _pending_keyframe_save is not None:
            _pending_keyframe_save["deferred"] = False
            _save_keyframe_snapshot(
                _pending_keyframe_save, raw_ext, raw_wrist, time.perf_counter() - rollout_start
            )
            _pending_keyframe_save = None

    # -----------------------------------------------------------------------
    # Execution thread: runs at FPS, fetches one action per step from the
    # shared buffer, drives the arm and gripper.
    # -----------------------------------------------------------------------
    def _execution_loop() -> None:
        nonlocal step, ended_by
        while step < args.max_episode_steps and not done_event.is_set():
            t0 = time.perf_counter()

            observation = get_observation()

            # Timestamp after get_observation(), which blocks on both cameras.
            if _video_mode:
                _capture_video_frame(time.perf_counter(), observation)

            cmd = _get_action(observation)

            arm_cmd = cmd[:6].copy()
            arm_cmd[3:6] = arm_cmd[3:6] / np.pi * 180
            if _gripper_open_override.is_set():
                arm.set_gripper_position(850)
            else:
                gripper_norm = float(cmd[6]) if cmd.shape[0] > 6 else 0.0
                if gripper_norm > args.gripper_close_threshold:
                    gripper_norm = args.gripper_close_snap
                arm.set_gripper_position(850 - gripper_norm * 860)

            pose = arm.get_position()[1]
            pose[3] = pose[3] % 360
            pose[5] = pose[5] % 360
            state = np.array(pose, dtype=np.float32)
            interpolate_action(state, arm_cmd)
            step += 1


            # Track how much of the budget the step actually consumed. If busy_s
            # regularly exceeds DT the loop is running below FPS, which silently
            # stretches anything measured in steps rather than seconds.
            busy_s = time.perf_counter() - t0
            exec_busy.append(busy_s)
            time.sleep(max(DT - busy_s, 0))
            exec_periods.append(time.perf_counter() - t0)

        if done_event.is_set():
            ended_by = "user"
        done_event.set()

    # --- initialize action buffer with the first inference ---
    _observation_curr = get_observation()
    if _feed_text_history:
        _observation_curr["phase_history"] = _format_event_history([])

    init_obs = dict(_observation_curr)  # mutable copy for potential extra keys
    if args.video_encoder and args.mode in ("text_keyframe", "keyframe", "no_memory", "text"):
        # Signal the policy server to reset its hidden-state cache. no_memory/text seed once
        # with zero history and never slide (no reset needed after this) — matches
        # always-zero-padded semantics without ever resending/recomputing the pad.
        # Video stride mode doesn't need caching since frames come at fixed intervals.
        init_obs["reset_cache"] = True
    elif _video_mode:
        # Seed the zero-filled grid, then take the t=0 capture so the grid is anchored at
        # rollout start. First stack is therefore [zeros..., frame@0, frame@0].
        _init_video_buffer(init_obs)
        _capture_video_frame(time.perf_counter(), init_obs)
        with _video_lock:
            init_obs["observation/exterior_image_1_left"] = np.stack(
                list(_video_buf_ext) + [init_obs["observation/exterior_image_1_left"]], axis=0,
            )
            init_obs["observation/wrist_image_left"] = np.stack(
                list(_video_buf_wrist) + [init_obs["observation/wrist_image_left"]], axis=0,
            )
    elif args.video_encoder:
        init_obs = _build_keyframe_obs(init_obs, [], [], args.num_frames)
    t0_init = time.perf_counter()
    first_output = policy.infer(init_obs)
    infer_latencies.append(time.perf_counter() - t0_init)
    first_actions = np.array(first_output["actions"], dtype=np.float32)
    if first_actions.ndim == 3:
        first_actions = first_actions[0]
    n = min(first_actions.shape[0], PREDICTION_HORIZON)
    _action_curr[:n, :] = first_actions[:n, :7]

    # --- launch threads ---
    infer_thread = threading.Thread(target=_inference_loop, daemon=True)
    exec_thread = threading.Thread(target=_execution_loop, daemon=True)
    infer_thread.start()
    exec_thread.start()

    try:
        exec_thread.join()
    except KeyboardInterrupt:
        ended_by = "keyboard_interrupt"
        logging.info("Rollout %d interrupted at step %d", rollout_idx + 1, step)
    finally:
        done_event.set()

    infer_thread.join(timeout=2.0)

    total_time = time.perf_counter() - rollout_start
    with _event_lock:
        completed_events = list(_completed_event_chunks)
        event_history_final = _format_event_history(_completed_event_chunks)
        keyframe_records = list(_keyframe_records)

    if args.save_keyframes and keyframe_records:
        _keyframe_img_dir.mkdir(parents=True, exist_ok=True)
        (_keyframe_img_dir / "metadata.json").write_text(json.dumps({
            "rollout_idx": rollout_idx,
            "task": args.task,
            "mode": args.mode,
            "prompt": args.prompt,
            "steps": step,
            "ended_by": ended_by,
            "event_history": event_history_final,
            "images": keyframe_records,
        }, indent=2) + "\n")
        logging.info(
            "Saved %d events (%d images) + metadata.json to %s",
            len(keyframe_records), 2 * len(keyframe_records), _keyframe_img_dir,
        )

    lats = infer_latencies
    achieved_hz = (1.0 / float(np.mean(exec_periods))) if exec_periods else 0.0
    overruns = sum(1 for b in exec_busy if b > DT)
    logging.info(
        "Rollout %d done | ended_by=%s | steps=%d | total=%.1fs | "
        "inferences=%d | mean_infer=%.3fs | events=%s",
        rollout_idx + 1, ended_by, step, total_time,
        len(lats), float(np.mean(lats)) if lats else 0.0, completed_events,
    )
    logging.info(
        "  exec loop: %.2f Hz achieved (target %.0f) | %d/%d steps overran DT",
        achieved_hz, FPS, overruns, len(exec_busy),
    )
    if exec_busy and overruns > 0.05 * len(exec_busy):
        logging.warning(
            "  execution loop missed its deadline on %.0f%% of steps — anything measured "
            "in steps rather than seconds is stretched by ~%.0f%%.",
            100.0 * overruns / len(exec_busy),
            100.0 * (FPS / achieved_hz - 1.0) if achieved_hz else 0.0,
        )
    return {
        "rollout_idx": rollout_idx,
        "completed_events": completed_events,
        "event_history": event_history_final,
        "steps": step,
        "total_time_s": total_time,
        "num_inferences": len(lats),
        "mean_infer_s": float(np.mean(lats)) if lats else 0.0,
        "min_infer_s": float(np.min(lats)) if lats else 0.0,
        "max_infer_s": float(np.max(lats)) if lats else 0.0,
        "exec_hz": achieved_hz,
        "exec_overrun_frac": (overruns / len(exec_busy)) if exec_busy else 0.0,
        "ended_by": ended_by,
    }



def run_event_inference(args: Args) -> dict:
    """Set up hardware and run args.num_rollouts rollouts. Returns {"rollouts": [...]}."""
    _setup(args)

    run_dir = _make_run_out_dir(args.out_path)
    rollouts = []
    for i in range(args.num_rollouts):
        result = run_rollout(args, rollout_idx=i, out_dir=run_dir)
        rollouts.append(result)

    return {"rollouts": rollouts}


# ------------------------
# Main
# ------------------------

def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    run_event_inference(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
