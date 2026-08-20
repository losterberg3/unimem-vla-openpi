"""Generate event labels and event-history overlays for LIBERO-style kitchen episodes.

Per-task labeling functions detect transition events from the recorded
state (gripper signal and end-effector z), then label a small window around
each transition with the completed event id. The remaining timesteps stay at
``EVENT_LABEL_IGNORE`` (mapped to a dedicated logits class during training).

Kitchen tasks each have a dedicated labeler (``kitchen1`` … ``kitchen6``).
"""

import io
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2

import numpy as np
import pandas as pd
from PIL import Image

# Asymmetric window around each event transition (first index of new event = ``i``):
# ``[i - TRANSITION_BACK_STEPS, i + TRANSITION_FORWARD_STEPS]`` (clamped).
TRANSITION_BACK_STEPS = 25
TRANSITION_FORWARD_STEPS = 15

# Steps after a transition before the completion phrase appears in event_history.
HISTORY_DELAY_STEPS = 8

# Z-action treated as "zero" (settled / released) when |dz| <= this.
Z_ACTION_ZERO_TOLERANCE = 1e-6

# Write this value into the dataset ``labels`` column for timesteps without a semantic event target.
# Training maps ``-1`` to a dedicated logits class (see ``EVENT_LABEL_IGNORE_TARGET_CLASS`` in ``pi0.py``).
EVENT_LABEL_IGNORE = -1

# Short phrases appended when *exiting* each event (completed steps only).
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


EVENT_LABELS = [
    "0: Grabbed box",
    "1: Dropped left",
    "2: Dropped right",
    "3: Retracted",
    "4: Tapped left basket",
    "5: Tapped right basket",
    "6: Placed box",
    "7: Empty basket",
    "8: Found butter",
    "9: Got butter",
    "10: Tapped plate",
]


DEFAULT_DATASET_ROOT = "~/.cache/huggingface/lerobot/<your-hf-username>/mem4/data/chunk-000"
DEFAULT_TASK_NAME = "kitchen4"
DEFAULT_EPISODE_START = 0
DEFAULT_EPISODE_END_EXCLUSIVE = 100
DEFAULT_VIDEO_PREFIX = "task_debug"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_FPS = 10.0
DEFAULT_FRAME_WIDTH = 256
DEFAULT_FRAME_HEIGHT = 256
DEFAULT_HISTORY_MAX_CHARS = 500


def episode_path(dataset_root: str, episode_num: int) -> str:
    root = os.path.expanduser(dataset_root)
    return os.path.join(root, f"episode_{episode_num:06d}.parquet")


# ---------------------------------------------------------------------------
# Per-task labeling
# ---------------------------------------------------------------------------


def _apply_window_label(labels: list[int], trigger: int, event_id: int, back: int, forward: int) -> None:
    if trigger is None:
        return
    n = len(labels)
    start = max(0, trigger - back)
    end = min(n - 1, trigger + forward)
    for i in range(start, end + 1):
        labels[i] = event_id


def _build_event_history(
    n: int,
    events: list[tuple[int | None, int]],
    history_delay: int,
) -> list[str]:
    """events: list of (trigger_step, event_id).

    Each phrase becomes visible in event_history once ``step >= trigger + delay``.
    """
    activations: list[tuple[int, str]] = []
    for trigger, event_id in events:
        if trigger is None:
            continue
        base = COMPLETION_LANGUAGE.get(event_id)
        if base is None:
            continue
        activations.append((trigger + history_delay, base))
    activations.sort(key=lambda x: x[0])

    histories: list[str] = []
    next_idx = 0
    for i in range(n):
        while next_idx < len(activations) and i >= activations[next_idx][0]:
            next_idx += 1
        if next_idx == 0:
            histories.append("History: none")
            continue
        chunks: list[str] = []
        for j in range(next_idx):
            _, base = activations[j]
            chunks.append(base)
        histories.append("History: " + ", ".join(chunks))
    return histories


def label_kitchen1(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,
) -> tuple[list[int], list[str], list[int], bool]:
    """Label kitchen1 demos: grab box -> retract -> place box on plate.

    Event-id sequence: 0 (grabbed box), 3 (retracted), 6 (placed box).

    Detection (uses ``actions``, not ``state``: cleaner discrete signals)
    --------
    Action layout: ``[dx, dy, dz, drx, dry, drz, gripper_cmd]``;
    ``gripper_cmd`` is -1 (open) or +1 (close).

    * grab    : first step where gripper command transitions -1 -> +1.
    * retract : after grab, the lift apex - first step where dz turns negative
                after we have seen a strictly positive dz (i.e. lifting -> lowering).
    * place   : after retract, first step where ``|dz| <= z_zero_tol`` (motion settles
                back to zero, i.e. the box has been released).

    Returns
    -------
    labels         : list[int], length n. Default ``EVENT_LABEL_IGNORE``; event id within
                     ``[trigger - transition_back, trigger + transition_forward]``.
    event_history  : list[str], length n. Cumulative completion phrases delayed by
                     ``history_delay`` steps after each trigger.
    events_for_overlay : list[int], length n. Coarse current-event id useful for video overlay.
    """
    actions = np.stack(df["actions"].values)
    n = len(actions)
    dz = actions[:, 2]
    grip_cmd = actions[:, 6]

    grab_step: int | None = None
    retract_step: int | None = None
    place_step: int | None = None

    # Grab: gripper command flips from open (-) to closed (+).
    for i in range(1, n):
        if grip_cmd[i - 1] < 0 and grip_cmd[i] > 0:
            grab_step = i
            break

    # Retract: after grab, after seeing positive dz (lifting), first negative dz (lowering).
    if grab_step is not None:
        saw_positive = False
        for i in range(grab_step + 1, n):
            if dz[i] > 0:
                saw_positive = True
            elif saw_positive and dz[i] < 0:
                retract_step = i
                break

    # Place: after retract, first step where dz settles to zero (released).
    if retract_step is not None:
        for i in range(retract_step + 1, n):
            if abs(dz[i]) <= z_zero_tol:
                place_step = i
                break

    labels: list[int] = [EVENT_LABEL_IGNORE] * n
    _apply_window_label(labels, grab_step, 0, transition_back, transition_forward)
    _apply_window_label(labels, retract_step, 3, transition_back, transition_forward)
    _apply_window_label(labels, place_step, 6, transition_back, transition_forward)

    event_history = _build_event_history(
        n,
        events=[
            (grab_step, 0),
            (retract_step, 3),
            (place_step, 6),
        ],
        history_delay=history_delay,
    )

    # Coarse current-event id used purely for video overlay / debugging.
    events_for_overlay: list[int] = []
    for i in range(n):
        if place_step is not None and i >= place_step:
            events_for_overlay.append(6)  # placed box
        elif retract_step is not None and i >= retract_step:
            events_for_overlay.append(3)
        elif grab_step is not None and i >= grab_step:
            events_for_overlay.append(0)  # grabbed box
        else:
            events_for_overlay.append(EVENT_LABEL_IGNORE)

    print(
        f"  events: grab={grab_step} retract={retract_step} place={place_step} (n={n})"
    )
    return labels, event_history, events_for_overlay


def _retract_step_between_grab_and_place(dz: np.ndarray, grab: int, place: int) -> int | None:
    """Lift apex between close and open: require positive dz then first negative dz.

    Same criterion as single-cycle kitchen1/2 retract, but the search is bounded to
    ``(grab, place)`` so later cycles cannot steal the signal.
    """
    if place <= grab + 1:
        return None
    saw_positive = False
    for i in range(grab + 1, place):
        if dz[i] > 0:
            saw_positive = True
        elif saw_positive and dz[i] < 0:
            return i
    return None


def _label_repeated_pick_place(
    df: pd.DataFrame,
    *,
    grab_event_ids: list[int],
    place_event_ids: list[int],
    retract_event_ids: list[int] | None = None,
    transition_back: int,
    transition_forward: int,
    history_delay: int,
) -> tuple[list[int], list[str], list[int], bool]:
    """Generic labeler for an alternating sequence of "grab object, place object" cycles.

    Detection: gripper-command transitions on actions[:, 6] (-1 = open, +1 = close):
    * grab_k  : k-th step where grip flips -1 -> +1, labeled ``grab_event_ids[k]``.
    * place_k : k-th step where grip flips +1 -> -1, labeled ``place_event_ids[k]``.
    * retract_k (optional): if ``retract_event_ids`` is set, same as single-cycle tasks —
      after ``grab_k``, once ``dz`` has been strictly positive (lift), the first step
      where ``dz < 0`` before ``place_k``.

    Number of cycles is ``min(len(grab_event_ids), len(place_event_ids))``. Pass a
    repeated list (e.g. ``[0]*3``) for tasks that reuse one (grab,place) pair.
    Pass distinct ids per cycle (e.g. ``[4, 6]``/``[5, 7]``) for tasks that grab
    different objects across cycles.
    """
    actions = np.stack(df["actions"].values)
    n = len(actions)
    grip_cmd = actions[:, 6]
    dz = actions[:, 2]

    num_cycles = min(len(grab_event_ids), len(place_event_ids))

    grab_steps: list[int] = []
    place_steps: list[int] = []
    for i in range(1, n):
        if grip_cmd[i - 1] < 0 and grip_cmd[i] > 0:
            grab_steps.append(i)
        elif grip_cmd[i - 1] > 0 and grip_cmd[i] < 0:
            place_steps.append(i)
    grab_steps = grab_steps[:num_cycles]
    place_steps = place_steps[:num_cycles]

    retract_steps: list[int | None] = []
    if retract_event_ids is not None:
        n_retract = min(num_cycles, len(retract_event_ids), len(grab_steps), len(place_steps))
        for k in range(n_retract):
            retract_steps.append(
                _retract_step_between_grab_and_place(dz, grab_steps[k], place_steps[k])
            )

    labels: list[int] = [EVENT_LABEL_IGNORE] * n
    events: list[tuple[int | None, int]] = []
    overlay_pairs: list[tuple[int, int]] = []
    for k, s in enumerate(grab_steps):
        pid = grab_event_ids[k]
        _apply_window_label(labels, s, pid, transition_back, transition_forward)
        events.append((s, pid))
        overlay_pairs.append((s, pid))
    if retract_event_ids is not None:
        for k, s in enumerate(retract_steps):
            if s is None:
                continue
            pid = retract_event_ids[k]
            _apply_window_label(labels, s, pid, transition_back, transition_forward)
            events.append((s, pid))
            overlay_pairs.append((s, pid))
    for k, s in enumerate(place_steps):
        pid = place_event_ids[k]
        _apply_window_label(labels, s, pid, transition_back, transition_forward)
        events.append((s, pid))
        overlay_pairs.append((s, pid))
    events.sort(key=lambda x: x[0] if x[0] is not None else -1)

    event_history = _build_event_history(
        n,
        events=events,
        history_delay=history_delay,
    )

    overlay_pairs.sort(key=lambda x: x[0])
    events_for_overlay: list[int] = []
    cur = EVENT_LABEL_IGNORE
    next_idx = 0
    for i in range(n):
        while next_idx < len(overlay_pairs) and i >= overlay_pairs[next_idx][0]:
            cur = overlay_pairs[next_idx][1]
            next_idx += 1
        events_for_overlay.append(cur)

    print(f"  events: grabs={grab_steps} retracts={retract_steps} places={place_steps} (n={n})")
    return labels, event_history, events_for_overlay


def label_kitchen2(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001  (signature uniformity)
) -> tuple[list[int], list[str], list[int], bool]:
    """Lift the box and place it back on the plate, 2 times. Events: 0 (grab box), 6 (place box)."""
    return _label_repeated_pick_place(
        df,
        grab_event_ids=[0] * 4,
        place_event_ids=[6] * 4,
        retract_event_ids=[3] * 4,
        transition_back=transition_back,
        transition_forward=transition_forward,
        history_delay=history_delay,
    )

def label_kitchen3(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001
    drop_side: str = "left",
    tap_side: str = "left",
) -> tuple[list[int], list[str], list[int]]:
    """Grab box, drop it left or right, retract, then tap a basket at the end of the demo.

    Events: 0 (grabbed box), 1/2 (dropped left/right), 3 (retracted), 4/5 (tapped left/right basket).

    drop_side: "left" -> event 1, "right" -> event 2
    tap_side:  "left" -> event 4, "right" -> event 5

    Tap label is applied to the last ``transition_back`` frames (no gripper signal; fixed end window).
    """
    drop_event_id = 1 if drop_side == "left" else 2
    tap_event_id = 4 if tap_side == "left" else 5

    actions = np.stack(df["actions"].values)
    ee_pos = np.stack(df["state"].values)[:, :3]
    n = len(actions)
    grip_cmd = actions[:, 6]

    grab_step: int | None = None
    drop_step: int | None = None
    retract_step: int | None = None

    # Grab: gripper flips open -> closed
    for i in range(1, n):
        if grip_cmd[i - 1] < 0 and grip_cmd[i] > 0:
            grab_step = i
            break

    # Drop: after grab, gripper flips closed -> open
    if grab_step is not None:
        for i in range(grab_step + 1, n):
            if grip_cmd[i - 1] > 0 and grip_cmd[i] < 0:
                drop_step = i
                break

    # Retract: after drop, minimum EE X (furthest back in -x direction)
    if drop_step is not None and drop_step < n - 1:
        seg_x = ee_pos[drop_step + 1:, 0]
        retract_step = int(drop_step + 1 + int(np.argmin(seg_x)))

    # Tap: ~20 frames before end of demo
    tap_step = max(0, n - 20)

    labels: list[int] = [EVENT_LABEL_IGNORE] * n
    _apply_window_label(labels, grab_step, 0, transition_back, transition_forward)
    _apply_window_label(labels, drop_step, drop_event_id, transition_back, transition_forward)
    _apply_window_label(labels, retract_step, 3, transition_back, transition_forward)
    _apply_window_label(labels, tap_step, tap_event_id, transition_back, 0)

    events = [
        (grab_step, 0),
        (drop_step, drop_event_id),
        (retract_step, 3),
        (tap_step, tap_event_id),
    ]
    event_history = _build_event_history(
        n,
        events=events,
        history_delay=history_delay,
    )

    ordered = sorted(
        [(s, pid) for s, pid in [
            (grab_step, 0), (drop_step, drop_event_id), (retract_step, 3), (tap_step, tap_event_id)
        ] if s is not None],
        key=lambda x: x[0],
    )
    events_for_overlay: list[int] = []
    cur = EVENT_LABEL_IGNORE
    next_idx = 0
    for i in range(n):
        while next_idx < len(ordered) and i >= ordered[next_idx][0]:
            cur = ordered[next_idx][1]
            next_idx += 1
        events_for_overlay.append(cur)

    print(f"  events: grab={grab_step} drop={drop_step} retract={retract_step} tap@{tap_step} (n={n})")
    return labels, event_history, events_for_overlay


def label_kitchen4(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001
) -> tuple[list[int], list[str], list[int], bool]:
    """Pick up box, retract, place it back. Events: 0 (grabbed box), 3 (retracted), 6 (placed box)."""
    return _label_repeated_pick_place(
        df,
        grab_event_ids=[0],
        place_event_ids=[6],
        retract_event_ids=[3],
        transition_back=transition_back,
        transition_forward=transition_forward,
        history_delay=history_delay,
    )


def label_kitchen5(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001
    rot_stop_threshold: float = 0.01,
    rot_search_window: int = 40,
) -> tuple[list[int], list[str], list[int]]:
    """Find butter at max EE Y, then got butter when arm stops rotating before gripper opens.

    Events: 8 (found butter) at argmax(ee_y), 9 (got butter) when rotation stops in the
    window immediately before the gripper opens (~2 s at 10 Hz).
    """
    actions = np.stack(df["actions"].values)
    ee_pos = np.stack(df["state"].values)[:, :3]
    n = len(actions)

    # Found butter: step where EE Y is most positive
    found_butter_step = int(np.argmax(ee_pos[:, 1]))

    grip_cmd = actions[:, 6]

    # Find the gripper-open event after found_butter (anchor for the rotation-stop search)
    grip_open_step: int | None = None
    for i in range(found_butter_step + 1, n):
        if grip_cmd[i - 1] > 0 and grip_cmd[i] < 0:
            grip_open_step = i
            break

    # Got butter: last step where rotation was active in the window before gripper opens,
    # plus one (i.e. the first step after rotation has stopped).
    got_butter_step: int | None = None
    if grip_open_step is not None:
        window_start = max(found_butter_step, grip_open_step - rot_search_window)
        rot_norms = np.linalg.norm(actions[window_start:grip_open_step, 3:6], axis=1)
        active = np.where(rot_norms > rot_stop_threshold)[0]
        if len(active) > 0:
            got_butter_step = window_start + int(active[-1]) + 1
        else:
            got_butter_step = grip_open_step

    labels: list[int] = [EVENT_LABEL_IGNORE] * n
    _apply_window_label(labels, found_butter_step, 8, transition_back, transition_forward)
    _apply_window_label(labels, got_butter_step, 9, transition_back, transition_forward)

    events = [
        (found_butter_step, 8),
        (got_butter_step, 9),
    ]
    event_history = _build_event_history(
        n,
        events=events,
        history_delay=history_delay,
    )

    ordered = sorted(
        [(s, pid) for s, pid in [(found_butter_step, 8), (got_butter_step, 9)] if s is not None],
        key=lambda x: x[0],
    )
    events_for_overlay: list[int] = []
    cur = EVENT_LABEL_IGNORE
    next_idx = 0
    for i in range(n):
        while next_idx < len(ordered) and i >= ordered[next_idx][0]:
            cur = ordered[next_idx][1]
            next_idx += 1
        events_for_overlay.append(cur)

    print(f"  events: found_butter={found_butter_step} got_butter={got_butter_step} grip_open={grip_open_step} (n={n})")
    return labels, event_history, events_for_overlay

def label_kitchen5_part2(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001
    retract_y_tol: float = 0.02,
    rot_stop_threshold: float = 0.01,
    rot_search_window: int = 40,
) -> tuple[list[int], list[str], list[int]]:
    """Empty basket at max EE Y, retracted when back to start Y, found butter at min EE Y, got butter at rotation stop.

    Events: 7 (empty basket) at argmax(ee_y), 3 (retracted) when ee_y returns within
    retract_y_tol of the starting Y, 8 (found butter) at argmin(ee_y) after retract,
    9 (got butter) when rotation stops in the window before gripper opens.
    """
    actions = np.stack(df["actions"].values)
    ee_pos = np.stack(df["state"].values)[:, :3]
    n = len(actions)
    grip_cmd = actions[:, 6]

    start_y = ee_pos[0, 1]

    # Empty basket: step where EE Y is most positive
    empty_basket_step = int(np.argmax(ee_pos[:, 1]))

    # Retracted: after empty_basket, first step where ee_y is within retract_y_tol of start_y
    retract_step: int | None = None
    for i in range(empty_basket_step + 1, n):
        if abs(ee_pos[i, 1] - start_y) <= retract_y_tol:
            retract_step = i
            break

    # Found butter: argmin(ee_y) after retract (or after empty_basket if no retract found)
    search_start = retract_step if retract_step is not None else empty_basket_step
    found_butter_step = int(search_start + np.argmin(ee_pos[search_start:, 1]))

    # Gripper-open event after found_butter (anchor for rotation-stop search)
    grip_open_step: int | None = None
    for i in range(found_butter_step + 1, n):
        if grip_cmd[i - 1] > 0 and grip_cmd[i] < 0:
            grip_open_step = i
            break

    # Got butter: first step after rotation stops in window before gripper opens
    got_butter_step: int | None = None
    if grip_open_step is not None:
        window_start = max(found_butter_step, grip_open_step - rot_search_window)
        rot_norms = np.linalg.norm(actions[window_start:grip_open_step, 3:6], axis=1)
        active = np.where(rot_norms > rot_stop_threshold)[0]
        if len(active) > 0:
            got_butter_step = window_start + int(active[-1]) + 1
        else:
            got_butter_step = grip_open_step

    labels: list[int] = [EVENT_LABEL_IGNORE] * n
    _apply_window_label(labels, empty_basket_step, 7, transition_back, transition_forward)
    _apply_window_label(labels, retract_step, 3, transition_back, transition_forward)
    _apply_window_label(labels, found_butter_step, 8, transition_back, transition_forward)
    _apply_window_label(labels, got_butter_step, 9, transition_back, transition_forward)

    events = [
        (empty_basket_step, 7),
        (retract_step, 3),
        (found_butter_step, 8),
        (got_butter_step, 9),
    ]
    event_history = _build_event_history(
        n,
        events=events,
        history_delay=history_delay,
    )

    ordered = sorted(
        [(s, pid) for s, pid in [
            (empty_basket_step, 7), (retract_step, 3),
            (found_butter_step, 8), (got_butter_step, 9),
        ] if s is not None],
        key=lambda x: x[0],
    )
    events_for_overlay: list[int] = []
    cur = EVENT_LABEL_IGNORE
    next_idx = 0
    for i in range(n):
        while next_idx < len(ordered) and i >= ordered[next_idx][0]:
            cur = ordered[next_idx][1]
            next_idx += 1
        events_for_overlay.append(cur)

    print(
        f"  events: empty_basket={empty_basket_step} retract={retract_step} "
        f"found_butter={found_butter_step} got_butter={got_butter_step} grip_open={grip_open_step} (n={n})"
    )
    return labels, event_history, events_for_overlay


def label_kitchen6(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001
) -> tuple[list[int], list[str], list[int]]:
    """Got butter on first gripper close; tapped plate at lowest EE Z in last 30 frames.

    Events: 9 (got butter) at first grip open->close, 10 (tapped plate) at argmin(ee_z)
    in the last 30 frames.
    """
    actions = np.stack(df["actions"].values)
    ee_pos = np.stack(df["state"].values)[:, :3]
    n = len(actions)
    grip_cmd = actions[:, 6]

    # Got butter: first gripper close
    got_butter_step: int | None = None
    for i in range(1, n):
        if grip_cmd[i - 1] < 0 and grip_cmd[i] > 0:
            got_butter_step = i
            break

    # Tapped plate: lowest EE Z in last 30 frames
    tail_start = max(0, n - 30)
    tapped_plate_step = int(tail_start + int(np.argmin(ee_pos[tail_start:, 2])))

    labels: list[int] = [EVENT_LABEL_IGNORE] * n
    _apply_window_label(labels, got_butter_step, 9, transition_back, transition_forward)
    _apply_window_label(labels, tapped_plate_step, 10, transition_back, transition_forward)

    events = [
        (got_butter_step, 9),
        (tapped_plate_step, 10),
    ]
    event_history = _build_event_history(
        n,
        events=events,
        history_delay=history_delay,
    )

    ordered = sorted(
        [(s, pid) for s, pid in [(got_butter_step, 9), (tapped_plate_step, 10)] if s is not None],
        key=lambda x: x[0],
    )
    events_for_overlay: list[int] = []
    cur = EVENT_LABEL_IGNORE
    next_idx = 0
    for i in range(n):
        while next_idx < len(ordered) and i >= ordered[next_idx][0]:
            cur = ordered[next_idx][1]
            next_idx += 1
        events_for_overlay.append(cur)

    print(f"  events: got_butter={got_butter_step} tapped_plate={tapped_plate_step} (n={n})")
    return labels, event_history, events_for_overlay

# Map task names to per-task labeling functions. Add more as we author them.
TASK_LABELERS = {
    "kitchen1": label_kitchen1,
    "kitchen2": label_kitchen2,
    "kitchen3": label_kitchen3,
    "kitchen4": label_kitchen4,
    "kitchen5": label_kitchen5,
    "kitchen5_part2": label_kitchen5_part2,
    "kitchen6": label_kitchen6,
}


# ---------------------------------------------------------------------------
# Video overlay helpers
# ---------------------------------------------------------------------------


def wrap_text(text: str, max_chars: int = 70) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > max_chars and cur:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return lines


def draw_overlay(
    frame: np.ndarray,
    *,
    current_event: int,
    event_history_text: str,
    target_label: int,
    dz_action: float,
    grip_cmd: float,
    history_max_chars: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    display_event = EVENT_LABELS[current_event] if 0 <= current_event < len(EVENT_LABELS) else "(none)"
    hist_lines = wrap_text(event_history_text, max_chars=history_max_chars)

    small_h = 16
    big_h = 22
    pad = 8
    n_hist = max(1, len(hist_lines))
    block_h = pad + big_h + n_hist * small_h + small_h + pad

    y0 = h - block_h
    cv2.rectangle(frame, (5, y0), (w - 5, h - 5), (0, 0, 0), -1)

    y = y0 + pad + 16
    cv2.putText(
        frame,
        f"Event: {display_event}",
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y += big_h

    for hl in hist_lines:
        cv2.putText(
            frame,
            hl,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += small_h

    cv2.putText(
        frame,
        f"Event target: {target_label}",
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"dz:{dz_action:+.3f} grip_cmd:{grip_cmd:+.1f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 255),
        1,
    )
    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Generate event labels and debug videos for kitchen episodes.")
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--task-name",
        type=str,
        default=DEFAULT_TASK_NAME,
        help=f"Which per-task labeler to use. Available: {sorted(TASK_LABELERS)}",
    )
    parser.add_argument("--episode-start", type=int, default=DEFAULT_EPISODE_START)
    parser.add_argument("--episode-end", type=int, default=DEFAULT_EPISODE_END_EXCLUSIVE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-prefix", type=str, default=DEFAULT_VIDEO_PREFIX)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--history-max-chars", type=int, default=DEFAULT_HISTORY_MAX_CHARS)
    parser.add_argument("--transition-back", type=int, default=TRANSITION_BACK_STEPS)
    parser.add_argument("--transition-forward", type=int, default=TRANSITION_FORWARD_STEPS)
    parser.add_argument("--history-delay", type=int, default=HISTORY_DELAY_STEPS)
    parser.add_argument("--z-zero-tol", type=float, default=Z_ACTION_ZERO_TOLERANCE)
    parser.add_argument(
        "--drop-side",
        choices=["left", "right"],
        default="left",
        help="kitchen3: side the box is dropped into. left=event1, right=event2.",
    )
    parser.add_argument(
        "--tap-side",
        choices=["left", "right"],
        default="left",
        help="kitchen3: basket tapped at the end of the demo. left=event4, right=event5.",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write labels / phase_history back to parquet, then patch meta/info.json "
        "& episodes_stats.jsonl like lerobot_meta_add_features (unless --no-lerobot-meta-update).",
    )
    parser.add_argument(
        "--lerobot-repo-root",
        type=str,
        default="",
        help=(
            "LeRobot dataset root (folder containing meta/ and data/). "
            "If empty, inferred from --dataset-root (e.g. parent of data/chunk-000)."
        ),
    )
    parser.add_argument(
        "--no-lerobot-meta-update",
        action="store_true",
        help="With --write-parquet, skip updating meta/info.json and meta/episodes_stats.jsonl.",
    )
    return parser


def resolve_lerobot_repo_root(dataset_root: str, explicit_repo_root: str | None = None) -> Path:
    """Find repo root containing ``meta/info.json`` for LeRobot-format datasets."""
    if explicit_repo_root and explicit_repo_root.strip():
        root = Path(os.path.expanduser(explicit_repo_root.strip())).resolve()
        info = root / "meta" / "info.json"
        if not info.is_file():
            raise FileNotFoundError(f"--lerobot-repo-root {root} does not contain meta/info.json")
        return root

    chunk_dir = Path(os.path.expanduser(dataset_root)).resolve()
    candidate = chunk_dir / "meta" / "info.json"
    if candidate.is_file():
        return chunk_dir

    if chunk_dir.name.startswith("chunk-") and chunk_dir.parent.name == "data":
        root = chunk_dir.parent.parent
        if (root / "meta" / "info.json").is_file():
            return root

    if chunk_dir.name == "data":
        root = chunk_dir.parent
        if (root / "meta" / "info.json").is_file():
            return root

    raise FileNotFoundError(
        f"Cannot infer LeRobot repo root from dataset-root {chunk_dir}: "
        "expected .../<repo>/data/chunk-* or repo with meta/info.json; pass --lerobot-repo-root."
    )


def _update_lerobot_metadata_after_parquet_writes(repo_root: Path) -> None:
    """Register labels / phase_history in meta/info.json and meta/episodes_stats.jsonl."""
    _NEW_FEATURES = {
        "labels": {"dtype": "int64", "shape": [1], "names": None},
        "phase_history": {"dtype": "string", "shape": [1], "names": None},
    }
    _NUMERIC_NEW = ["labels"]  # string columns have no numeric stats

    print("Updating LeRobot dataset metadata...")

    # --- update meta/info.json ---
    info_path = repo_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    added = []
    for name, spec in _NEW_FEATURES.items():
        if name not in info["features"]:
            info["features"][name] = spec
            added.append(name)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    if added:
        print(f"  info.json: added features {added}")
    else:
        print("  info.json: features already present, no change")

    # --- update meta/episodes_stats.jsonl ---
    stats_path = repo_root / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        print("  episodes_stats.jsonl not found, skipping stats update")
        return

    with open(stats_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # Prefer the template from info.json to locate parquets
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunks_size = info.get("chunks_size", 1000)

    updated_entries = []
    for entry in entries:
        ep_idx = entry["episode_index"]
        ep_chunk = ep_idx // chunks_size
        parquet_rel = data_template.format(episode_chunk=ep_chunk, episode_index=ep_idx)
        parquet_path = repo_root / parquet_rel
        if not parquet_path.exists():
            updated_entries.append(entry)
            continue

        df = pd.read_parquet(parquet_path)
        ep_stats = entry.get("stats", {})
        for col in _NUMERIC_NEW:
            if col not in df.columns:
                continue
            vals = np.stack(df[col].values).astype(np.float64)  # (N,) or (N, D)
            if vals.ndim == 1:
                vals = vals[:, None]
            ep_stats[col] = {
                "min": vals.min(axis=0).tolist(),
                "max": vals.max(axis=0).tolist(),
                "mean": vals.mean(axis=0).tolist(),
                "std": vals.std(axis=0).tolist(),
                "count": [len(vals)],
            }
        entry["stats"] = ep_stats
        updated_entries.append(entry)

    with open(stats_path, "w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  episodes_stats.jsonl: updated {len(updated_entries)} episodes")


def main() -> None:
    args = parse_args().parse_args()

    if args.task_name not in TASK_LABELERS:
        raise ValueError(
            f"No labeling function registered for task '{args.task_name}'. "
            f"Available: {sorted(TASK_LABELERS)}"
        )
    labeler = TASK_LABELERS[args.task_name]

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using task profile: {args.task_name}")

    wrote_parquet_any = False
    for episode_num in range(args.episode_start, args.episode_end):
        parquet_path = episode_path(args.dataset_root, episode_num)
        if not os.path.exists(parquet_path):
            continue

        df = pd.read_parquet(parquet_path)
        print(f"[ep {episode_num:06d}] n={len(df)}")

        labeler_kwargs: dict = dict(
            transition_back=args.transition_back,
            transition_forward=args.transition_forward,
            history_delay=args.history_delay,
            z_zero_tol=args.z_zero_tol,
        )
        if args.task_name == "kitchen3":
            labeler_kwargs["drop_side"] = args.drop_side
            labeler_kwargs["tap_side"] = args.tap_side
        labels, event_history_values, events_for_overlay = labeler(df, **labeler_kwargs)

        if args.write_parquet:
            df["labels"] = np.asarray(labels, dtype=np.int64)
            df["phase_history"] = event_history_values
            df.to_parquet(parquet_path, index=False)
            wrote_parquet_any = True
            print(f"  wrote labels, phase_history -> {parquet_path}")
        else:
            print("  dry-run (use --write-parquet to persist)")

        frames: list[np.ndarray] = []
        for i in range(len(df)):
            row = df.iloc[i]
            img = Image.open(io.BytesIO(row["image"]["bytes"]))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            action = np.asarray(row["actions"], dtype=np.float64)
            frame = draw_overlay(
                frame,
                current_event=events_for_overlay[i],
                event_history_text=event_history_values[i],
                target_label=labels[i],
                dz_action=float(action[2]),
                grip_cmd=float(action[6]),
                history_max_chars=args.history_max_chars,
            )
            frames.append(frame)

        video_name = f"{args.video_prefix}_{episode_num:06d}.mp4"
        video_path = output_dir / video_name
        out = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.frame_width, args.frame_height),
        )
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"  saved {video_path}")

    if (
        args.write_parquet
        and wrote_parquet_any
        and not args.no_lerobot_meta_update
    ):
        try:
            repo = resolve_lerobot_repo_root(
                args.dataset_root,
                explicit_repo_root=args.lerobot_repo_root.strip() or None,
            )
            _update_lerobot_metadata_after_parquet_writes(repo)
        except FileNotFoundError as err:
            print(f"Skipping LeRobot meta update: {err}")


if __name__ == "__main__":
    main()
