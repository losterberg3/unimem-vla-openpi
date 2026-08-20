#!/usr/bin/env python3
"""Run LIBERO inference.

Queries the policy server, executes actions in the LIBERO environment, and saves
a robot-view video. Five modes control memory usage: no_memory, text,
text_keyframe, keyframe, video (see Args.mode).
"""

import collections
import dataclasses
import logging
from typing import Literal
import os
import pathlib
import sys
import time

# Local LIBERO checkout (no pip install): add the *repo root* (where setup.py lives), same as
# LIBERO/scripts/init_path.py (../ from scripts). That layout exposes `libero.libero`, not .../LIBERO/libero alone.
def _prepend_libero_repo_path() -> None:
    if os.environ.get("LIBERO_SUPPRESS_LOCAL_PATH"):
        return
    extra = os.environ.get("LIBERO_REPO_PATH", os.environ.get("LIBERO_PACKAGE_PATH", "")).strip()
    candidates: list[pathlib.Path] = []
    if extra:
        candidates.append(pathlib.Path(extra).expanduser().resolve())
    here = pathlib.Path(__file__).resolve()
    # openpi at msl/openpi → sibling msl/LIBERO
    candidates.append(here.parents[3] / "LIBERO")
    candidates.append(pathlib.Path.home() / "msl" / "LIBERO")

    for root in candidates:
        inner = root / "libero" / "libero" / "__init__.py"
        if inner.is_file():
            root_s = str(root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
            return


_prepend_libero_repo_path()

import imageio.v2 as imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image, ImageDraw, ImageFont
import tyro

LIBERO_ENV_RESOLUTION = 256

# Event head constants.
_NUM_EVENT_CLASSES = 11
_EVENT_IGNORE_CLASS_ID = 11
_EVENT_CONFIDENCE_THRESHOLD = 0.8

# Phrases appended to event history on event transitions (matches label_dataset.py).
COMPLETION_LANGUAGE = {
    0: "grabbed box",
    1: "dropped left",
    2: "dropped right",
    3: "retracted",
    4: "tapped left basket",
    5: "tapped right basket",
    6: "placed box",
    7: "empty basket",
    8: "found butter",
    9: "got butter",
    10: "tapped plate",
}


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle. Matches official LIBERO/robosuite convention."""
    import math
    q = quat.copy()
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def _predicted_event_from_probs(probs: np.ndarray, threshold: float = _EVENT_CONFIDENCE_THRESHOLD) -> int:
    """Argmax of event-class probabilities (already softmaxed in pi0); return ignore class if below threshold."""
    event_probs = np.asarray(probs, dtype=np.float64).reshape(-1)[:_NUM_EVENT_CLASSES]
    k = int(np.argmax(event_probs))
    if float(event_probs[k]) > threshold:
        return k
    return _EVENT_IGNORE_CLASS_ID


def _format_event_history(completed_chunks: list[str], *, label: str = "History") -> str:
    history = ", ".join(completed_chunks)
    return f"{label}: {history}" if history else f"{label}: none"


# --- Environment helpers ---

class SimpleVisualizationWrapper:
    """Enables robot/gripper visualization overlays (only needed when training used them)."""

    def __init__(self, env):
        self.env = env
        self._vis_settings = {"env": True, "robots": True, "grippers": False}
        self._inner_env = None
        e = env
        for _ in range(10):
            if hasattr(e, 'visualize'):
                self._inner_env = e
                break
            if hasattr(e, 'env'):
                e = e.env
            else:
                break
        if self._inner_env is None:
            logging.warning("SimpleVisualizationWrapper: could not find env with visualize() method")
        self._update_visualization()

    def _update_visualization(self):
        if self._inner_env is not None:
            self._inner_env.visualize(vis_settings=self._vis_settings)

    def reset(self):
        obs = self.env.reset()
        self._update_visualization()
        return obs

    def step(self, action):
        result = self.env.step(action)
        self._update_visualization()
        return result

    def seed(self, seed):
        return self.env.seed(seed)

    @property
    def sim(self):
        return self.env.sim

    def check_success(self):
        return self.env.check_success()

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()


def annotate_frame(img: np.ndarray, text: str, step: int) -> np.ndarray:
    """Add a text annotation bar above the frame."""
    original_img = Image.fromarray(img)
    img_width, img_height = original_img.size

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    if ']' in text:
        parts = text.split(']', 1)
        prim_num = parts[0] + ']'
        prim_desc = parts[1].strip()
    else:
        prim_num = ""
        prim_desc = text

    temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    max_width = img_width - 20

    words = prim_desc.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = temp_draw.textbbox((0, 0), test_line, font=font_large)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    line_height = temp_draw.textbbox((0, 0), "Ay", font=font_large)[3] - temp_draw.textbbox((0, 0), "Ay", font=font_large)[1]
    padding = 8
    text_area_height = (line_height * 3) + (padding * 2) + 5

    new_height = img_height + text_area_height
    new_img = Image.new('RGB', (img_width, new_height), color=(30, 30, 40))
    new_img.paste(original_img, (0, text_area_height))

    draw = ImageDraw.Draw(new_img)
    y_pos = padding
    draw.text((padding, y_pos), prim_num, fill=(255, 200, 0), font=font_large)
    y_pos += line_height + 2
    for line in lines:
        draw.text((padding, y_pos), line, fill=(255, 255, 255), font=font_large)
        y_pos += line_height

    step_text = f"Step {step}"
    step_bbox = draw.textbbox((0, 0), step_text, font=font_small)
    step_width = step_bbox[2] - step_bbox[0]
    step_height = step_bbox[3] - step_bbox[1]
    step_padding = 6
    step_x = img_width - step_width - step_padding * 2
    step_y = new_height - step_height - step_padding * 2
    draw.rectangle(
        [step_x - step_padding, step_y - step_padding, img_width - step_padding, new_height - step_padding],
        fill=(0, 0, 0, 200),
        outline=(100, 100, 100),
    )
    draw.text((step_x, step_y), step_text, fill=(200, 200, 200), font=font_small)

    # libx264 requires macroblock height divisible by 16.
    nh = new_img.height
    if nh % 16 != 0:
        pad = 16 - (nh % 16)
        taller = Image.new("RGB", (new_img.width, nh + pad), color=(30, 30, 40))
        taller.paste(new_img, (0, 0))
        new_img = taller

    return np.array(new_img)


# --- Args and main entry point ---

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    max_episode_steps: int = 1000
    num_rollouts: int = 1
    video_out_path: str = "data/libero/videos"
    save_video: bool = True
    run_name: str = ""
    batch_name: str = ""
    checkpoint: str = ""
    # Task: either benchmark (suite + index) or an explicit BDDL path.
    bddl_file: str = ""
    task_suite_name: str = "libero_mem"
    task_index: int = 0
    # If non-empty, sent to the policy as `prompt` instead of the task's LIBERO language string.
    prompt_override: str = ""
    match_demo_collection_env: bool = True
    controller: str = "OSC_POSE"
    render_camera: str = "agentview"
    ignore_done: bool = True
    # Set True only if training used robosuite visualization overlays.
    visualization_overlays: bool = False
    seed: int = -1
    # True: send `phase_history` and use `event_id` transitions.
    # False: plain prompt-only rollout (for non-event checkpoints).
    # If True, print raw model action chunks at each replan.
    log_raw_actions: bool = False
    infer_every_steps: int = 10
    # Confidence threshold for event head predictions.
    event_confidence_threshold: float = _EVENT_CONFIDENCE_THRESHOLD
    # Video encoder: model was trained with video_encoder=True and expects (T,H,W,C) images.
    # Must match the checkpoint's training config.
    video_encoder: bool = False
    num_frames: int = 4
    # Inference mode — controls what memory is sent to the model each step:
    #   "no_memory"    — no event history text, no historical frames (prompt + current frame only).
    #   "text"         — event history text only; zeros for video history slots.
    #   "text_keyframe"   — event history text + keyframes always; requires video_encoder=True.
    #   "keyframe"        — keyframes always, NO event history text; requires video_encoder=True.
    #   "video" — rolling stride-based frames every stride_steps steps, NO text;
    #                    requires video_encoder=True.
    mode: Literal["no_memory", "text", "text_keyframe", "keyframe", "video"] = "text"
    # Steps between frame captures in video_stride mode.
    stride_steps: int = 20
    # When set to an obs key (e.g. "box_a_pos"), records the object's XY position
    # at the start and end of each rollout for object_return_xy scoring.
    track_object_pos_key: str = ""


def run_event_inference(args: Args) -> None:
    """Run one or more LIBERO rollouts and optionally save videos."""
    from datetime import timezone, timedelta

    pst = timezone(timedelta(hours=-8))

    logging.info("event_confidence_threshold=%.4f", args.event_confidence_threshold)

    if args.seed < 0:
        args.seed = np.random.randint(0, 10000)
        logging.info(f"Using random seed: {args.seed}")
    np.random.seed(args.seed)

    if args.bddl_file:
        bddl_path = pathlib.Path(args.bddl_file).expanduser().resolve()
        if not bddl_path.is_file():
            raise FileNotFoundError(f"BDDL not found: {bddl_path}")
        import libero.libero.envs.bddl_utils as BDDLUtils
        problem_info = BDDLUtils.get_problem_info(str(bddl_path))
        task_prompt = str(problem_info.get("language_instruction", ""))
    else:
        benchmark_dict = benchmark.get_benchmark_dict()
        if args.task_suite_name not in benchmark_dict:
            raise ValueError(f"Unknown task suite: {args.task_suite_name}")
        task_suite = benchmark_dict[args.task_suite_name]()
        if not (0 <= args.task_index < task_suite.n_tasks):
            raise ValueError(f"task_index {args.task_index} out of range for suite {args.task_suite_name}")
        task = task_suite.get_task(args.task_index)
        task_prompt = str(task.language)
        bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    if args.prompt_override.strip():
        task_prompt = args.prompt_override.strip()

    env_kwargs = dict(
        bddl_file_name=str(bddl_path),
        robots=["Panda"],
        controller=args.controller,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
        camera_names=["agentview", "robot0_eye_in_hand"],
        render_camera=args.render_camera,
        control_freq=20,
        horizon=3000,
    )
    if args.match_demo_collection_env:
        env_kwargs["ignore_done"] = args.ignore_done

    env = OffScreenRenderEnv(**env_kwargs)
    if args.visualization_overlays:
        env = SimpleVisualizationWrapper(env)
    env.seed(args.seed)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    rollouts: list[dict] = []

    try:
        for rollout_idx in range(args.num_rollouts):
            if args.num_rollouts > 1:
                logging.info("=" * 60)
                logging.info("Rollout %d / %d", rollout_idx + 1, args.num_rollouts)
            completed, steps, event_history, chunks, total_time, avg_infer_time, initial_obj_xy, final_obj_xy, event_xy = (
                _run_rollout_after_env_ready(args, env, client, bddl_path, task_prompt, pst, rollout_idx)
            )
            rollout = {
                "event_history": event_history,
                "chunks": chunks,
                "completed": completed,
                "steps": steps,
                "total_time_s": round(total_time, 2),
                "avg_infer_ms": round(avg_infer_time * 1000, 1),
            }
            if initial_obj_xy is not None and final_obj_xy is not None:
                rollout["initial_obj_xy"] = initial_obj_xy.tolist()
                rollout["final_obj_xy"] = final_obj_xy.tolist()
            if event_xy:
                rollout["event_xy"] = event_xy
            rollouts.append(rollout)
    finally:
        try:
            env.close()
        except Exception:
            logging.debug("env.close() failed", exc_info=True)

    return {"rollouts": rollouts}


def _run_rollout_after_env_ready(
    args: Args,
    env,
    client,
    bddl_path: pathlib.Path,
    task_prompt: str,
    pst,
    rollout_idx: int = 0,
) -> tuple[bool, int, str, list[str]]:
    """Rollout loop, video, and logging. Returns (completed, steps, event_history_text, completed_event_chunks)."""
    from datetime import datetime

    timestamp = datetime.now(pst).strftime("%Y%m%d_%H%M%S")
    date_folder = datetime.now(pst).strftime("%Y-%m-%d")
    run_dir = args.run_name if args.run_name else f"run_{timestamp}"
    if args.batch_name:
        video_out_path = pathlib.Path(args.video_out_path) / date_folder / args.batch_name / run_dir
    else:
        video_out_path = pathlib.Path(args.video_out_path) / date_folder / run_dir
    video_out_path.mkdir(parents=True, exist_ok=True)
    logging.info("Run output directory: %s", video_out_path)

    server_metadata = client.get_server_metadata()
    checkpoint_config = server_metadata.get("checkpoint_config", "unknown")
    checkpoint_dir = server_metadata.get("checkpoint_dir", "unknown")
    checkpoint_info = args.checkpoint or f"{checkpoint_config} @ {checkpoint_dir}"

    metadata_file = video_out_path / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Checkpoint: {checkpoint_info}\n")
        f.write(f"Timestamp: {datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S')} PST\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Task: {task_prompt}\n")
        f.write(f"BDDL: {bddl_path}\n")
        if not args.bddl_file:
            f.write(f"Suite/index: {args.task_suite_name}/{args.task_index}\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")
    logging.info("Saved metadata to: %s", metadata_file)

    logging.info("=" * 60)
    logging.info("Task: %s", task_prompt)
    logging.info("=" * 60)

    obs = env.reset()
    # Open gripper: env.reset() gives qpos≈0 (closed); run several open steps so
    # qpos reaches the training-distribution open state (~0.04) before inference.
    for _ in range(10):
        obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # -1 = open in LIBERO Panda

    # Record initial object XY before the robot acts.
    initial_obj_xy: np.ndarray | None = None
    if args.track_object_pos_key and args.track_object_pos_key in obs:
        initial_obj_xy = np.asarray(obs[args.track_object_pos_key], dtype=np.float64)[:2].copy()
        logging.info("Initial object XY (%s): %.3f  %.3f", args.track_object_pos_key, *initial_obj_xy)

    # _event_tracking: detect events and capture keyframes (all modes that use the event head).
    # _use_text_history: actually include phase_history text in the observation.
    _event_tracking = args.mode in ("text", "text_keyframe", "keyframe")
    _use_text_history = args.mode in ("text", "text_keyframe")

    all_replay_images: list[np.ndarray] = []
    t = 0
    task_completed = False
    task_completion_step = None
    action_queue: collections.deque = collections.deque()
    latest_action_chunk: np.ndarray | None = None
    last_infer_step = -(10 ** 9)
    infer_times: list[float] = []
    episode_start_time = time.monotonic()

    # Event frame buffer (video_encoder=True only): holds num_frames-1 event snapshots.
    # Populated on event transitions; current frame appended at inference time.
    # Whether the buffer contents are sent depends on mode.
    _use_video_frames: bool = False
    if args.video_encoder:
        _init_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            args.resize_size, args.resize_size,
        ))
        _init_wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            args.resize_size, args.resize_size,
        ))
        _event_slots = args.num_frames - 1
        frame_buffer_img: collections.deque = collections.deque(
            [np.zeros_like(_init_img) for _ in range(_event_slots)], maxlen=_event_slots
        )
        frame_buffer_wrist: collections.deque = collections.deque(
            [np.zeros_like(_init_wrist) for _ in range(_event_slots)], maxlen=_event_slots
        )
        if args.mode == "video":
            stride_buffer_img: collections.deque = collections.deque(
                [np.zeros_like(_init_img)] * _event_slots, maxlen=_event_slots
            )
            stride_buffer_wrist: collections.deque = collections.deque(
                [np.zeros_like(_init_wrist)] * _event_slots, maxlen=_event_slots
            )
            _last_stride_step: int = -(args.stride_steps)

    event_history_text = "History: none"
    last_sent_event_history: str = event_history_text
    last_appended_event: int | None = None
    completed_event_chunks: list[str] = []

    event_log: list[tuple[int, str]] = []
    event_xy: dict[str, list[float]] = {}

    while t < args.max_episode_steps:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

        if args.mode == "video" and args.video_encoder and (t - _last_stride_step) >= args.stride_steps:
            stride_buffer_img.append(img.copy())
            stride_buffer_wrist.append(wrist_img.copy())
            _last_stride_step = t

        should_infer_now = (t == 0) or ((t - last_infer_step) >= args.infer_every_steps)
        if should_infer_now:
            if args.video_encoder:
                if args.mode == "video":
                    _use_video_frames = True
                else:
                    _use_video_frames = args.mode in ("text_keyframe", "keyframe")

                if args.mode == "video":
                    obs_img = np.stack(list(stride_buffer_img) + [img], axis=0)
                    obs_wrist = np.stack(list(stride_buffer_wrist) + [wrist_img], axis=0)
                elif _use_video_frames:
                    obs_img = np.stack(list(frame_buffer_img) + [img], axis=0)
                    obs_wrist = np.stack(list(frame_buffer_wrist) + [wrist_img], axis=0)
                else:
                    obs_img = np.stack([np.zeros_like(img)] * _event_slots + [img], axis=0)
                    obs_wrist = np.stack([np.zeros_like(wrist_img)] * _event_slots + [wrist_img], axis=0)
            else:
                _use_video_frames = False
                obs_img = img
                obs_wrist = wrist_img

            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ))
            element = {
                "observation/image": obs_img,
                "observation/wrist_image": obs_wrist,
                "observation/state": state,
                "prompt": task_prompt,
            }
            if _use_text_history:
                ph = _format_event_history(completed_event_chunks)
                # Wire key stays "phase_history": it must match the key name the served
                # checkpoint's data pipeline (and its LeRobot dataset column) was trained
                # with. Renaming it here would require also renaming it in the training
                # pipeline, which is out of scope for this terminology cleanup.
                element["phase_history"] = ph
                last_sent_event_history = ph

            _t0 = time.monotonic()
            infer_output = client.infer(element)
            _latency = time.monotonic() - _t0
            infer_times.append(_latency)
            last_infer_step = t

            raw_batch = np.asarray(infer_output["actions"])
            action_chunk = raw_batch
            if action_chunk.ndim == 3 and action_chunk.shape[0] > 0:
                action_chunk = action_chunk[0]

            predicted_event = -1
            if _event_tracking:
                predicted_event = _predicted_event_from_probs(
                    infer_output["event_id"],
                    threshold=args.event_confidence_threshold,
                )
                if predicted_event in COMPLETION_LANGUAGE and predicted_event != last_appended_event:
                    event_label = COMPLETION_LANGUAGE[predicted_event]
                    completed_event_chunks.append(event_label)
                    last_appended_event = predicted_event
                    event_log.append((t, event_label))
                    event_xy[event_label] = obs["robot0_eef_pos"][:2].tolist()
                    # Snapshot current frame into the event buffer on event transition.
                    if args.video_encoder and args.mode in ("text_keyframe", "keyframe"):
                        frame_buffer_img.append(img)
                        frame_buffer_wrist.append(wrist_img)
                event_history_text = _format_event_history(completed_event_chunks)
                logging.info(
                    "  Replan t=%d: event=%d | video=%s | tracked: %s | sent: %s",
                    t, predicted_event, _use_video_frames, event_history_text, last_sent_event_history,
                )
            else:
                logging.info("  Replan t=%d", t)

            if args.log_raw_actions:
                raw_chunk = np.asarray(action_chunk)
                head_n = min(5, int(raw_chunk.shape[0])) if raw_chunk.ndim > 0 else 0
                if head_n > 0:
                    logging.info(
                        "  Raw action chunk head (first %d):\n%s",
                        head_n,
                        np.array2string(raw_chunk[:head_n], precision=4, suppress_small=False),
                    )

            latest_action_chunk = np.asarray(action_chunk)

        # Annotate and record frame.
        if _event_tracking:
            overlay = f"[rollout] {task_prompt}, {_format_event_history(completed_event_chunks)}"
        else:
            overlay = f"[rollout] {task_prompt}"
        all_replay_images.append(annotate_frame(img, overlay, t))

        # Execution cadence: commit a new chunk only when the queue drains.
        if not action_queue:
            if latest_action_chunk is None:
                raise RuntimeError("No inferred action chunk available to execute.")
            commit_n = min(args.infer_every_steps, int(latest_action_chunk.shape[0]))
            for a in latest_action_chunk[:commit_n]:
                action_queue.append(a)

        action = np.asarray(action_queue.popleft(), dtype=np.float64).copy()

        try:
            obs, _, done, _ = env.step(action.tolist())
            done = False
        except ValueError as e:
            if "terminated episode" in str(e):
                logging.info("  Episode terminated at step %d", t)
                done = True
            else:
                raise

        if done:
            if not task_completed:
                task_completed = True
                task_completion_step = t
                logging.info("  Task completed at step %d", t)
            t += 1
            break

        t += 1

    # Save video.
    video_path: pathlib.Path | None = None
    if args.save_video and all_replay_images:
        video_fname = f"rollout_{rollout_idx:03d}_{timestamp}.mp4" if args.num_rollouts > 1 else f"rollout_{timestamp}.mp4"
        video_path = video_out_path / video_fname
        imageio.mimwrite(str(video_path), all_replay_images, fps=40)
        logging.info("Saved video: %s", video_path)
    else:
        logging.info("Skipped video save (--no-save-video).")

    total_time = time.monotonic() - episode_start_time
    avg_infer_time = float(np.mean(infer_times)) if infer_times else 0.0

    # Record final object XY.
    final_obj_xy: np.ndarray | None = None
    if args.track_object_pos_key and args.track_object_pos_key in obs:
        final_obj_xy = np.asarray(obs[args.track_object_pos_key], dtype=np.float64)[:2].copy()
        logging.info("Final object XY (%s): %.3f  %.3f", args.track_object_pos_key, *final_obj_xy)
        if initial_obj_xy is not None:
            dist = float(np.linalg.norm(final_obj_xy - initial_obj_xy))
            logging.info("Object XY displacement: %.4f m", dist)

    logging.info("=" * 60)
    logging.info("Result: %s", "completed" if task_completed else "not completed")
    if task_completed:
        logging.info("Completed at step: %d", task_completion_step)
    logging.info("Total steps: %d", t)
    logging.info("Total episode time: %.2f s", total_time)
    logging.info(
        "Inference calls: %d  |  avg=%.1f ms  min=%.1f ms  max=%.1f ms",
        len(infer_times),
        avg_infer_time * 1000,
        1000 * np.min(infer_times) if infer_times else 0.0,
        1000 * np.max(infer_times) if infer_times else 0.0,
    )
    logging.info("Final event history: %s", event_history_text)
    if video_path is not None:
        logging.info("Video: %s", video_path)
    logging.info("=" * 60)

    return task_completed, t, event_history_text, completed_event_chunks, total_time, avg_infer_time, initial_obj_xy, final_obj_xy, event_xy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_event_inference(tyro.cli(Args))
