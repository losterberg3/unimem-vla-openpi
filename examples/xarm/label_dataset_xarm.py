"""Generate event labels and event-history overlays for xArm *hardware* episodes.

Hardware analogue of ``label_dataset_libero.py`` (which targets the sim/LIBERO
kitchen datasets). It reads an existing LeRobot-format dataset (point
``--dataset-root`` at your own, e.g. ``<your-hf-username>/mem7``),
runs a per-task labeling function that detects transition events from the
recorded state / gripper / actions, and labels a small window around each
transition with the completed event id. Timesteps without a semantic event
target stay at ``EVENT_LABEL_IGNORE``.

Class scheme
------------
11 event classes (ids ``0`` … ``10``) plus 1 null class = **12 logits classes**.
The null class is the ``EVENT_LABEL_IGNORE`` sentinel (``-1``), which training
maps to a dedicated logits class (index 11); see ``EVENT_LABEL_IGNORE_TARGET_CLASS``
in ``pi0.py``.

Dataset differences vs. the sim script
--------------------------------------
* 20 fps (sim is 10).
* Three camera streams (``exterior_image_1_left``, ``exterior_image_2_left``,
  ``wrist_image_left``) at 240x320 instead of a single 256x256 ``image``.
* ``state`` is a 6-vector (EE pose: xyz + rpy); the gripper is a separate
  ``gripper_position`` column rather than ``actions[:, 6]``.
* ``actions`` is a 7-vector.

Per-task labelers live in ``TASK_LABELERS``: ``xarm7``, ``xarm8``, ``xarm9``, and
``xarm10``, one per recorded task. Copy one of these as a starting point for a
new task and fill in its event detection — see the "Per-task labeling" section
below for the shared function signature every labeler follows.
"""

import io
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image

# Asymmetric window around each event transition (first index of new event = ``i``):
# ``[i - TRANSITION_BACK_STEPS, i + TRANSITION_FORWARD_STEPS]`` (clamped).
# NOTE: hardware is 20 fps (sim was 10), so windows cover half the wall-clock time
# of the same value in label_dataset_libero.py. Bump these if you want similar durations.
TRANSITION_BACK_STEPS = 1
TRANSITION_FORWARD_STEPS = 50

# Steps after a transition before the completion phrase appears in phase_history.
# Set to TRANSITION_FORWARD_STEPS + 1 so message only appears AFTER label window ends
HISTORY_DELAY_STEPS = TRANSITION_FORWARD_STEPS + 1

# EE z is "settled" when its per-step change |z[i] - z[i-1]| <= this (mm/step).
# (actions/state positions are ABSOLUTE pose in mm, not deltas, so settle/retract
# detection uses np.diff of the absolute coordinate.) Starting point; tune per task.
Z_ACTION_ZERO_TOLERANCE = 0.5

# gripper_position is continuous ~0.01 (open) .. ~0.51 (closed) on mem7; larger =
# more closed. Default close/open threshold is the midpoint; calibrate per task.
GRIPPER_CLOSED_THRESHOLD = 0.25

# "attached to hammer" = first frame below ATTACH_Z_MAX_MM (mm) that begins a run of
# at least ATTACH_NEG_X_RUN consecutive -x steps (reaching down + pulling to the hammer).
# ATTACH_MIN_NEG_X_DROP_MM guards against the ~1mm -x settle-drift right after grab:
# the run must cover a real net -x displacement, not sub-mm jitter.
ATTACH_Z_MAX_MM = 280.0
ATTACH_NEG_X_RUN = 10
ATTACH_MIN_NEG_X_DROP_MM = 5.0

# attach / extend / retract all happen at the far -y working region (by the hammer).
# Constrain them to y <= global y-min (after grab) + this band (mm).
Y_NEAR_MIN_BAND_MM = 70.0

# Write this value into the dataset ``labels`` column for timesteps without a
# semantic event target. Training maps ``-1`` to the dedicated null logits class.
EVENT_LABEL_IGNORE = -1

# ---------------------------------------------------------------------------
# Class vocabulary (11 event classes + 1 null class = 12 logits classes)
# ---------------------------------------------------------------------------
NUM_EVENT_CLASSES = 11          # valid event ids: 0 .. 10
NUM_LOGITS_CLASSES = 12         # + 1 null class (EVENT_LABEL_IGNORE)

# Short phrases appended to phase_history when *exiting* (completing) each event.
# Populate as tasks are defined; keys must be in [0, NUM_EVENT_CLASSES - 1].
COMPLETION_LANGUAGE: dict[int, str] = {
    0: "grabbed tape measure",
    1: "attached to hammer",
    2: "extended tape",
    3: "retracted tape",
    4: "placed tape measure",
}


# Human-readable labels for the video overlay, derived from COMPLETION_LANGUAGE.
EVENT_LABELS: dict[int, str] = {pid: f"{pid}: {txt}" for pid, txt in COMPLETION_LANGUAGE.items()}


def _validate_event_id(event_id: int) -> None:
    if not (0 <= event_id < NUM_EVENT_CLASSES):
        raise ValueError(
            f"event id {event_id} out of range [0, {NUM_EVENT_CLASSES - 1}] "
            f"({NUM_EVENT_CLASSES} event classes + 1 null class)"
        )


# ---------------------------------------------------------------------------
# Dataset accessors (hardware column layout)
# ---------------------------------------------------------------------------


def episode_path(dataset_root: str, episode_num: int) -> str:
    root = os.path.expanduser(dataset_root)
    return os.path.join(root, f"episode_{episode_num:06d}.parquet")


def ee_pose(df: pd.DataFrame) -> np.ndarray:
    """state -> (n, 6) EE pose [x, y, z, rx, ry, rz]; xyz in mm, rxyz in rad."""
    return np.stack(df["state"].values).astype(np.float64)


def ee_pos(df: pd.DataFrame) -> np.ndarray:
    """state[:, :3] -> (n, 3) EE position in mm."""
    return ee_pose(df)[:, :3]


def gripper(df: pd.DataFrame) -> np.ndarray:
    """gripper_position -> (n,) 1-D gripper signal (~0.01 open .. ~0.51 closed)."""
    return np.stack(df["gripper_position"].values).astype(np.float64).reshape(-1)


def actions(df: pd.DataFrame) -> np.ndarray:
    """actions -> (n, 7). ABSOLUTE target pose [x, y, z, rx, ry, rz, gripper]
    (not deltas): actions[:, :3] mm, [:, 3:6] rad, [:, 6] gripper (mirrors
    gripper_position). Use np.diff for velocities."""
    return np.stack(df["actions"].values).astype(np.float64)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def gripper_transitions(
    grip: np.ndarray,
    closed_threshold: float = GRIPPER_CLOSED_THRESHOLD,
    closed_is_high: bool = True,
) -> tuple[list[int], list[int]]:
    """Detect gripper close/open events from the continuous ``gripper_position`` signal.

    Crosses ``closed_threshold`` to define a boolean "closed" state, then returns
    the step indices where it transitions open->closed (grabs) and closed->open
    (releases).

    closed_is_high : if True, ``grip >= closed_threshold`` means closed (the mem7
        convention: larger values = more closed). Flip if your gripper encodes the
        opposite. Calibrate ``closed_threshold`` per task if 0.25 mislabels.
    """
    closed = grip >= closed_threshold if closed_is_high else grip <= closed_threshold
    close_steps: list[int] = []
    open_steps: list[int] = []
    for i in range(1, len(grip)):
        if not closed[i - 1] and closed[i]:
            close_steps.append(i)
        elif closed[i - 1] and not closed[i]:
            open_steps.append(i)
    return close_steps, open_steps


def gripper_fully_closed_step(
    grip: np.ndarray,
    close_start: int,
    plateau_lookahead: int = 60,
    settle_tol: float = 0.01,
) -> int:
    """First step at/after ``close_start`` where the gripper is FULLY closed.

    ``close_start`` is the threshold crossing where closing *begins* (e.g. from
    ``gripper_transitions``). The gripper then ramps to a closed plateau; this
    returns the first step within ``settle_tol`` of that plateau (the plateau is
    the max grip within ``plateau_lookahead`` steps of the crossing), i.e. when the
    grasp has finished closing rather than when it started.
    """
    n = len(grip)
    end = min(n, close_start + plateau_lookahead)
    plateau = float(grip[close_start:end].max()) if end > close_start else float(grip[close_start])
    for i in range(close_start, n):
        if grip[i] >= plateau - settle_tol:
            return i
    return close_start


def _retract_step_between(dz: np.ndarray, start: int, end: int) -> int | None:
    """Lift apex within (start, end): first negative dz after a strictly positive dz.

    ``dz`` is a per-step velocity, e.g. ``np.diff(ee_pos(df)[:, 2], prepend=...)``
    (positions are absolute mm, not deltas)."""
    if end <= start + 1:
        return None
    saw_positive = False
    for i in range(start + 1, end):
        if dz[i] > 0:
            saw_positive = True
        elif saw_positive and dz[i] < 0:
            return i
    return None


def smooth(v: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving-average smoothing (edge-shrinking 'same' convolution)."""
    if window <= 1:
        return v
    pad = window // 2
    vp = np.pad(v, pad, mode="edge")  # replicate edges (mode="same" zero-pads, corrupting ends)
    return np.convolve(vp, np.ones(window) / window, mode="valid")[: len(v)]


def first_rise_peak_after(
    series: np.ndarray,
    start: int,
    min_rise: float = 15.0,
    drop_tol: float = 8.0,
) -> int:
    """Index of the apex of the FIRST upward excursion in ``series`` after ``start``.100%

    Walks forward tracking the running max; once the value has risen at least
    ``min_rise`` above ``series[start]`` and then falls back by ``drop_tol`` from
    that running max, returns the running-max index (the apex). If no such peak
    exists (monotonic rise to the end), returns the global argmax — which for this
    dataset means the motion was still going when the recording stopped.

    Robust to the homing/return motions that produce a *second*, later apex:
    those are ignored because we stop at the first completed up-then-down peak.
    """
    n = len(series)
    if start >= n - 1:
        return start
    base = series[start]
    peak_val = series[start]
    peak_idx = start
    risen = False
    for i in range(start + 1, n):
        if series[i] > peak_val:
            peak_val = series[i]
            peak_idx = i
        if peak_val - base >= min_rise:
            risen = True
        if risen and series[i] <= peak_val - drop_tol:
            return peak_idx
    return peak_idx


def first_large_rise_onset_after(
    series: np.ndarray,
    start: int,
    min_rise: float = 20.0,
    base_band: float = 10.0,
) -> int:
    """Index where a large (>= ``min_rise``) upward run BEGINS after ``start``.

    Anchors on the apex of the first large rise (``first_rise_peak_after``), then
    returns the last step at/before that apex whose value is within ``base_band``
    of the climb's valley — the base of the ascent, i.e. when the value *starts*
    going up by a large amount, rather than the apex it eventually reaches.
    """
    apex = first_rise_peak_after(series, start, min_rise=min_rise)
    if apex <= start:
        return start
    seg = series[start:apex + 1]
    valley = float(seg.min())
    near = np.where(seg <= valley + base_band)[0]
    return start + int(near[-1])


def first_high_x_low_z(
    x: np.ndarray,
    z: np.ndarray,
    start: int,
    band: float,
    gap: int = 10,
) -> int:
    """Lowest-z index within the FIRST high-x run after ``start``.

    "High-x" = x within ``band`` mm of the post-``start`` max. The first contiguous
    run (bridging gaps <= ``gap``) is the initial reach toward the hammer; the
    lowest z inside it is where the arm presses down to hook on — i.e. high x AND
    low z. Taking the *first* run (not the global min over all high-x frames) keeps
    attach at the early reach, not a later one at the same x.
    """
    xseg = x[start:]
    hx = np.where(xseg >= xseg.max() - band)[0]
    if len(hx) == 0:
        return start
    run_end = hx[0]
    for k in hx[1:]:
        if k - run_end <= gap:
            run_end = k
        else:
            break
    run = np.arange(hx[0], run_end + 1)
    return start + int(run[np.argmin(z[start:][run])])


def _apply_window_label(labels: list[int], trigger: int | None, event_id: int, back: int, forward: int) -> None:
    if trigger is None:
        return
    _validate_event_id(event_id)
    n = len(labels)
    start = max(0, trigger - back)
    end = min(n - 1, trigger + forward)
    for i in range(start, end + 1):
        labels[i] = event_id


def _build_event_history(
    n: int,
    events: list[tuple[int | None, int]],
    history_delay: int,
    completion_language: dict[int, str] | None = None,
    forward_steps: dict[int, int] | None = None,
) -> list[str]:
    """events: list of (trigger_step, event_id).

    Each phrase becomes visible once ``step >= trigger + calculated_delay``.
    calculated_delay = forward_steps[event_id] + 1 if forward_steps provided, else history_delay.

    completion_language: per-task override of the global COMPLETION_LANGUAGE dict.
    forward_steps: dict mapping event_id to its forward labeling window size.
    """
    if completion_language is None:
        completion_language = COMPLETION_LANGUAGE
    activations: list[tuple[int, str, int]] = []
    for trigger, event_id in events:
        if trigger is None:
            continue
        base = completion_language.get(event_id)
        if base is None:
            continue
        # Use event-specific forward steps if provided, otherwise use global history_delay
        if forward_steps and event_id in forward_steps:
            delay = forward_steps[event_id] + 1
        else:
            delay = history_delay
        activations.append((trigger + delay, base, event_id))
    activations.sort(key=lambda x: x[0])

    histories: list[str] = []
    next_idx = 0
    for i in range(n):
        while next_idx < len(activations) and i >= activations[next_idx][0]:
            next_idx += 1
        visible = activations[:next_idx]
        if not visible:
            histories.append("History: none")
            continue
        histories.append("History: " + ", ".join(base for _, base, _ in visible))
    return histories


def _events_for_overlay(n: int, events: list[tuple[int | None, int]]) -> list[int]:
    """Coarse current-event id per step (latest triggered event), for video overlay."""
    ordered = sorted([(s, pid) for s, pid in events if s is not None], key=lambda x: x[0])
    events: list[int] = []
    cur = EVENT_LABEL_IGNORE
    next_idx = 0
    for i in range(n):
        while next_idx < len(ordered) and i >= ordered[next_idx][0]:
            cur = ordered[next_idx][1]
            next_idx += 1
        events.append(cur)
    return events


# ---------------------------------------------------------------------------
# Per-task labeling
# ---------------------------------------------------------------------------
#
# Each labeler has the signature:
#
#     def label_<task>(
#         df, *, transition_back, transition_forward, history_delay, z_zero_tol,
#     ) -> tuple[list[int], list[str], list[int]]
#
# returning (labels, event_history, events_for_overlay), each length n.
# See label_xarm7/8/9/10 below for real examples.


def label_xarm7(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001  (signature uniformity)
) -> tuple[list[int], list[str], list[int]]:
    """Measure the hammer with the tape measure (mem7 task 0).

    Event-id sequence (5 events):
      0 grabbed tape measure : gripper FULLY closed (reaches its closed plateau),
                               not the crossing where it starts to close.
    attach / extend / retract are all constrained to the far -y working region
    (y <= global y-min + Y_NEAR_MIN_BAND_MM), where the hammer is.

      1 attached to hammer    : first low-y frame below z=280 mm that begins >=10
                                consecutive -x steps (reach down and pull to the hammer).
      2 extended tape         : min x within the working region after attach (-x extreme).
      3 retracted tape        : onset of the upward (+z) move after extend (when the
                                arm starts moving up), within the working region.
      4 placed tape measure   : first gripper open (closed->open) after extend, if any
                                (most demos end holding the tape, so this is often absent).

    Detected independently rather than strictly chained at the tail, because the
    retract (+z) and place (gripper open) frequently coincide in the last frames.
    """
    n = len(df)
    pos = ee_pos(df)            # (n, 3) mm
    grip = gripper(df)          # (n,)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    sx = smooth(x)
    dx = np.diff(sx, prepend=sx[0])      # per-step x velocity (smoothed); < 0 = moving -x

    closes, opens = gripper_transitions(grip)

    # grab = gripper FULLY closed (closed plateau), not where it starts to close.
    grab_step: int | None = gripper_fully_closed_step(grip, closes[0]) if closes else None

    attach_step: int | None = None
    extend_step: int | None = None
    retract_step: int | None = None
    place_step: int | None = None

    # attach / extend / retract all occur at the far -y working region (by the hammer):
    # constrain them to y <= y_thr.
    y_thr = (float(y[grab_step:].min()) + Y_NEAR_MIN_BAND_MM) if grab_step is not None else 0.0

    # attach = first low-y frame below ATTACH_Z_MAX_MM that begins a run of at least
    # ATTACH_NEG_X_RUN consecutive -x steps covering a real net -x drop.
    if grab_step is not None:
        for i in range(grab_step, n - ATTACH_NEG_X_RUN):
            if (
                y[i] <= y_thr
                and z[i] < ATTACH_Z_MAX_MM
                and bool(np.all(dx[i + 1 : i + 1 + ATTACH_NEG_X_RUN] < 0))
                and x[i] - x[i + ATTACH_NEG_X_RUN] >= ATTACH_MIN_NEG_X_DROP_MM
            ):
                attach_step = i
                break

    # The working region ends when y first rises back above the band after attach.
    region_end = n - 1
    if attach_step is not None:
        region_end = next((k for k in range(attach_step, n) if y[k] > y_thr), n - 1)

    # extend = min x within the working region after attach (the -x extreme).
    if attach_step is not None and attach_step < region_end:
        extend_step = attach_step + int(np.argmin(x[attach_step : region_end + 1]))
    # retract = onset of the upward (+z) move after extend (arm starts moving up),
    # within the working region.
    if extend_step is not None and extend_step < region_end:
        retract_step = first_large_rise_onset_after(smooth(z)[: region_end + 1], extend_step)
        place_step = next((o for o in opens if o > extend_step), None)   # release, if any

    events: list[tuple[int | None, int]] = [
        (grab_step, 0),
        (attach_step, 1),
        (extend_step, 2),
        (retract_step, 3),
        (place_step, 4),
    ]
    events.sort(key=lambda e: e[0] if e[0] is not None else -1)

    labels: list[int] = [EVENT_LABEL_IGNORE] * n
    for trigger, event_id in events:
        _apply_window_label(labels, trigger, event_id, transition_back, transition_forward)

    event_history = _build_event_history(
        n,
        events=events,
        history_delay=history_delay,
    )
    events_for_overlay = _events_for_overlay(n, [(s, pid) for s, pid in events])

    print(
        f"  events: grab={grab_step} attach={attach_step} extend={extend_step} "
        f"retract={retract_step} place={place_step} (n={n})"
    )
    return labels, event_history, events_for_overlay


MEM8_COMPLETION_LANGUAGE: dict[int, str] = {
    0: "picked up spoon",
    1: "scooped beans",
    2: "poured beans",
    3: "placed spoon",
}
MEM8_EVENT_LABELS: dict[int, str] = {pid: f"{pid}: {txt}" for pid, txt in MEM8_COMPLETION_LANGUAGE.items()}

TASK_EVENT_LABELS: dict[str, dict[int, str]] = {
    "xarm8": MEM8_EVENT_LABELS,
}


def label_xarm8(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001
) -> tuple[list[int], list[str], list[int]]:
    """Spoon scooping and pouring beans three times (mem8 task).

    Event-id sequence (4 distinct ids, 8 trigger events):
      0 picked up spoon : gripper reaches its fully-closed plateau
      1 scooped beans   : minimum pitch (most forward tilt) in each scoop zone (fires 3x)
      2 poured beans    : minimum roll (maximum tilt to pour) in each pour zone (fires 3x)
      3 placed spoon    : first gripper open after the third pour

    Events 1 and 2 reuse the same id across all three repetitions — the model sees the
    same class label each time, and event_history accumulates the repeated phrases.

    Detection signals:
    - Pour zones:  roll (rx) drops below 145 deg AND y < -200 mm. The arm tilts the spoon
                   far to the side to pour; three distinct roll valleys are detectable.
    - Scoop zones: between consecutive pour groups, pitch (ry) goes most negative
                   (arm tilts forward into the bean bowl) while y is in [-150, -10] mm
                   (not in the spoon-pickup area at y~+150 or the pour region at y<-200).
    """
    n = len(df)
    pose = ee_pose(df)          # (n, 6) [x, y, z, roll, pitch, yaw]
    grip = gripper(df)          # (n,)
    _, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
    rx, _ = pose[:, 3], pose[:, 4]  # roll, pitch (rad)

    closes, opens = gripper_transitions(grip)
    grab_step: int | None = gripper_fully_closed_step(grip, closes[0]) if closes else None

    # --- Detect 3 pour groups: regions where roll < 145 deg AND y < -200 mm ---
    ROLL_POUR_THRESHOLD = np.radians(145.0)
    Y_POUR_THRESHOLD = -200.0
    POUR_GAP = 30  # min steps between distinct pour groups

    pour_groups: list[tuple[int, int]] = []
    if grab_step is not None:
        pour_mask = (rx[grab_step:] < ROLL_POUR_THRESHOLD) & (y[grab_step:] < Y_POUR_THRESHOLD)
        pour_indices = grab_step + np.where(pour_mask)[0]
        if len(pour_indices) > 0:
            gs = int(pour_indices[0])
            for k in range(1, len(pour_indices)):
                if int(pour_indices[k]) - int(pour_indices[k - 1]) > POUR_GAP:
                    pour_groups.append((gs, int(pour_indices[k - 1])))
                    gs = int(pour_indices[k])
            pour_groups.append((gs, int(pour_indices[-1])))

    # --- For each pour group: detect pour (min roll) and preceding scoop (min pitch) ---
    # Scoop zone: y in [-150, -10] — excludes spoon pickup at y~+150 and pour zone at y<-200
    SCOOP_Y_LO = -150.0
    SCOOP_Y_HI = -10.0

    scoop_steps: list[int | None] = []
    pour_steps: list[int | None] = []
    prev = grab_step if grab_step is not None else 0

    SCOOP_COMPLETION_OFFSET = 30  # frames after min-z to mark scoop as done

    for gs, ge in pour_groups[:3]:
        # scoop: min-z in the scoop y-zone, then shift forward to when the arm is
        # clearly rising back out with the beans.
        search = np.arange(prev, min(gs, n))
        zone = search[(y[search] > SCOOP_Y_LO) & (y[search] < SCOOP_Y_HI)]
        if len(zone) > 0:
            min_z_step = int(zone[int(np.argmin(z[zone]))])
            scoop_steps.append(min(min_z_step + SCOOP_COMPLETION_OFFSET, gs - 1))
        else:
            scoop_steps.append(None)
        # pour: min roll within the group
        pour_steps.append(gs + int(np.argmin(rx[gs : ge + 1])))
        prev = ge

    # pad to 3 entries if fewer pour groups were found
    while len(scoop_steps) < 3:
        scoop_steps.append(None)
    while len(pour_steps) < 3:
        pour_steps.append(None)

    # place: first gripper open after last (detected) pour
    last_event = next((p for p in reversed(pour_steps) if p is not None), grab_step or 0)
    place_step: int | None = next((o for o in opens if o > last_event), None)

    events: list[tuple[int | None, int]] = [
        (grab_step,      0),
        (scoop_steps[0], 1),
        (pour_steps[0],  2),
        (scoop_steps[1], 1),
        (pour_steps[1],  2),
        (scoop_steps[2], 1),
        (pour_steps[2],  2),
        (place_step,     3),
    ]

    labels: list[int] = [EVENT_LABEL_IGNORE] * n
    for trigger, event_id in events:
        _apply_window_label(labels, trigger, event_id, transition_back, transition_forward)

    event_history = _build_event_history(
        n,
        events=events,
        history_delay=history_delay,
        completion_language=MEM8_COMPLETION_LANGUAGE,
    )
    events_for_overlay = _events_for_overlay(n, [(s, pid) for s, pid in events])

    print(
        f"  events: grab={grab_step} "
        f"s1={scoop_steps[0]} p1={pour_steps[0]} "
        f"s2={scoop_steps[1]} p2={pour_steps[1]} "
        f"s3={scoop_steps[2]} p3={pour_steps[2]} "
        f"place={place_step} (n={n})"
    )
    return labels, event_history, events_for_overlay


MEM9_COMPLETION_LANGUAGE: dict[int, str] = {
    0: "grabbed bottle",
    1: "grabbed sponge",
    2: "wiped table",
    3: "placed sponge",
}
MEM9_EVENT_LABELS: dict[int, str] = {pid: f"{pid}: {txt}" for pid, txt in MEM9_COMPLETION_LANGUAGE.items()}

TASK_EVENT_LABELS["xarm9"] = MEM9_EVENT_LABELS


# mem10: one tap, one grab, one scoop, one pour, one place. Every event occurs
# exactly once, so there is no repetition to disambiguate and no counting to learn.
# This also removes the shortcut that made the retired 3-tap/3-pour mem10 scheme
# unlearnable: with a single pour, no cup ever contains beans while the arm is
# approaching, so the wrist view cannot identify the target by its contents and the
# tap keyframe is the only cue.
MEM10_COMPLETION_LANGUAGE: dict[int, str] = {
    0: "human tap",
    1: "grabbed spoon",
    2: "scooped beans",
    3: "poured beans",
    4: "placed spoon",
}
MEM10_EVENT_LABELS: dict[int, str] = {
    pid: f"{pid}: {txt}" for pid, txt in MEM10_COMPLETION_LANGUAGE.items()
}

TASK_EVENT_LABELS["xarm10"] = MEM10_EVENT_LABELS

# Tap label window, in frames either side of the human_event marker. The retired
# 3-tap/3-pour mem10 scheme used +-1
# (3 frames = 0.15s), which starved the tap in three separate ways: the classifier got
# 52x less supervision than scoop, the keyframe sampler drew from 3 near-identical
# frames so there was no augmentation, and at rollout the detector fired outside the
# window entirely. Widen it to cover the span where the hand is actually resting on the
# cup. 41 frames also saturates event_frame_window=30, so the sampler gets its full
# quota of distinct candidates. TUNE THESE to the hold duration actually collected.
MEM10_TAP_BACK = -5
MEM10_TAP_FORWARD = 31

# The scoop detector fires on the START of the 40-frame window over which pitch is
# straightening, so the anchor is systematically EARLY by up to that much — the spoon is
# still coming up out of the beans. Add frames here to slide the anchor toward the real
# scoop moment. This shifts the label window, the keyframe candidates, the upsample
# anchor and the history activation together, which asymmetric label windows do not:
# `_build_event_history` reads the raw event step, not the window.
# 0 disables the shift (anchor stays at the raw detected step). Pick a value by
# checking the saved overlay videos and moving it until the marker lands where the
# spoon actually lifts; ~20
# (half the detect window) is the natural starting guess.
MEM10_SCOOP_ANCHOR_OFFSET = 30


def label_xarm9(
    df: pd.DataFrame,
    transition_back: int = TRANSITION_BACK_STEPS,
    transition_forward: int = TRANSITION_FORWARD_STEPS,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001
) -> tuple[list[int], list[str], list[int]]:
    """Grab bottle, grab sponge, wipe table, place sponge (mem9 task).

    Event-id sequence (4 events):
      0 grabbed bottle : gripper reaches its fully-closed plateau on the first close
      1 grabbed sponge : gripper reaches its fully-closed plateau on the second close
      2 wiped table    : global minimum z after grabbing the sponge (arm pressed down
                         onto the table surface)
      3 placed sponge  : first gripper open after the wipe
    """
    n = len(df)
    pose = ee_pose(df)   # (n, 6) [x, y, z, roll, pitch, yaw]
    grip = gripper(df)   # (n,)
    z = pose[:, 2]

    closes, opens = gripper_transitions(grip)

    grab_bottle_step: int | None = (
        gripper_fully_closed_step(grip, closes[0]) if len(closes) >= 1 else None
    )
    grab_sponge_step: int | None = (
        gripper_fully_closed_step(grip, closes[1]) if len(closes) >= 2 else None
    )

    # wipe: global min z after the sponge is grabbed (arm pressed onto table)
    wipe_step: int | None = None
    if grab_sponge_step is not None:
        wipe_step = grab_sponge_step + int(np.argmin(z[grab_sponge_step:]))

    # place: first gripper open after the wipe
    place_step: int | None = None
    if wipe_step is not None:
        place_step = next((o for o in opens if o > wipe_step), None)

    events: list[tuple[int | None, int]] = [
        (grab_bottle_step, 0),
        (grab_sponge_step, 1),
        (wipe_step,        2),
        (place_step,       3),
    ]

    labels: list[int] = [EVENT_LABEL_IGNORE] * n
    for trigger, event_id in events:
        _apply_window_label(labels, trigger, event_id, transition_back, transition_forward)

    event_history = _build_event_history(
        n,
        events=events,
        history_delay=history_delay,
        completion_language=MEM9_COMPLETION_LANGUAGE,
    )
    events_for_overlay = _events_for_overlay(n, [(s, pid) for s, pid in events])

    print(
        f"  events: grab_bottle={grab_bottle_step} grab_sponge={grab_sponge_step} "
        f"wipe={wipe_step} place={place_step} (n={n})"
    )
    return labels, event_history, events_for_overlay


def label_xarm10(
    df: pd.DataFrame,
    transition_back: int = 5,
    transition_forward: int = 50,
    history_delay: int = HISTORY_DELAY_STEPS,
    z_zero_tol: float = Z_ACTION_ZERO_TOLERANCE,  # noqa: ARG001
    episode_num: int | None = None,  # noqa: ARG001
) -> tuple[list[int], list[str], list[int]]:
    """Fill ONE human-indicated cup with one scoop of beans (mem10 task).

    Event-id sequence (5 events, each occurring exactly once):
      0 human tap     : human_event == 1.0 at first occurrence
      1 grabbed spoon : gripper reaches its fully-closed plateau, first close after the tap
      2 scoop         : pitch straightening while z < 255 mm, roll/yaw steady
      3 pour          : minimum roll (furthest tilt) 150-450 frames after the scoop
      4 placed spoon  : first gripper open after the scoop

    Detection geometry matches the retired 3-tap/3-pour mem10 scheme (see git history)
    — same thresholds, same windows — so labels stay comparable across the two
    datasets. The only differences are that every event is taken once, the manual
    per-episode scoop overrides are dropped (they were specific to episodes in the
    old recording), and the tap gets a wide label
    window (see MEM10_TAP_BACK/FORWARD).
    """
    n = len(df)
    grip = gripper(df)
    closes, opens = gripper_transitions(grip)

    # --- Event 0: the human tap -------------------------------------------------
    tap_step: int | None = None
    if "human_event" in df.columns:
        hits = np.nonzero(np.asarray(df["human_event"].values, dtype=float) == 1.0)[0]
        if len(hits):
            tap_step = int(hits[0])
        if len(hits) > 1:
            print(f"  WARNING: {len(hits)} human_event markers; mem10 expects 1 (using the first)")

    events: list[tuple[int | None, int]] = [(tap_step, 0)]

    grab_spoon_step: int | None = None
    scoop_step: int | None = None
    pour_step: int | None = None
    place_step: int | None = None

    # --- Event 1: grabbed spoon (first full close after the tap) ----------------
    grab_closes = [c for c in closes if tap_step is not None and c > tap_step]
    if grab_closes:
        grab_spoon_step = gripper_fully_closed_step(grip, grab_closes[0])
        events.append((grab_spoon_step, 1))

    if grab_spoon_step is not None:
        pose = ee_pose(df)
        z_pos = pose[:, 2]
        roll_deg = np.degrees(pose[:, 3])
        yaw_deg = np.degrees(pose[:, 5])
        pitch_smooth = smooth(np.degrees(pose[:, 4]), window=5)
        dpitch = np.diff(pitch_smooth, prepend=pitch_smooth[0])

        SCOOP_DETECT_WINDOW = 40
        Z_SCOOP_THRESHOLD = 255.0

        # --- Event 2: scoop — sustained pitch straightening while low in the bowl
        for i in range(grab_spoon_step + 10, max(grab_spoon_step + 11, n - SCOOP_DETECT_WINDOW - 5)):
            window_roll = roll_deg[i : i + SCOOP_DETECT_WINDOW]
            if window_roll.std() > 5.0:
                continue
            roll_mean = window_roll.mean()
            if not ((160 <= roll_mean <= 200) or (-200 <= roll_mean <= -160)):
                continue
            if yaw_deg[i : i + SCOOP_DETECT_WINDOW].std() > 5.0:
                continue
            if np.mean(z_pos[i : i + SCOOP_DETECT_WINDOW]) > Z_SCOOP_THRESHOLD:
                continue
            if np.sum(dpitch[i : i + SCOOP_DETECT_WINDOW] > 0) < SCOOP_DETECT_WINDOW * 0.6:
                continue
            # `i` is the start of the straightening window, i.e. systematically early;
            # slide it toward the real scoop moment. Clamped so it stays in range.
            scoop_step = min(i + MEM10_SCOOP_ANCHOR_OFFSET, n - 1)
            break  # single scoop: stop at the first match

        if scoop_step is not None:
            events.append((scoop_step, 2))

            # --- Event 3: pour — furthest roll tilt after the scoop -------------
            search_start = scoop_step + 150
            search_end = min(scoop_step + 450, n)
            if search_end > search_start:
                pour_step = search_start + int(np.argmin(roll_deg[search_start:search_end]))
                events.append((pour_step, 3))

            # --- Event 4: placed spoon — first gripper open after the scoop -----
            place_step = next((o for o in opens if o > scoop_step), None)
            events.append((place_step, 4))
        else:
            place_step = next((o for o in opens if o > grab_spoon_step), None)
            if place_step is not None:
                events.append((place_step, 4))

    labels: list[int] = [EVENT_LABEL_IGNORE] * n

    # The tap gets a deliberately wide window (see MEM10_TAP_BACK/FORWARD); every
    # other window matches the retired 3-tap scheme exactly.
    if tap_step is not None:
        _apply_window_label(labels, tap_step, 0, MEM10_TAP_BACK, MEM10_TAP_FORWARD)
    if grab_spoon_step is not None:
        _apply_window_label(labels, grab_spoon_step, 1, transition_back - 5, transition_forward - 30)
    if scoop_step is not None:
        _apply_window_label(labels, scoop_step, 2, transition_back - 30, transition_forward + 10)
    if pour_step is not None:
        _apply_window_label(labels, pour_step, 3, transition_back + 15, transition_forward - 30)
    if place_step is not None:
        _apply_window_label(labels, place_step, 4, transition_back, transition_forward - 30)

    forward_steps_map = {
        0: MEM10_TAP_FORWARD,
        1: transition_forward - 30,
        2: transition_forward + 10,
        3: transition_forward - 30,
        4: transition_forward - 30,
    }

    event_history = _build_event_history(
        n,
        events=events,
        history_delay=history_delay,
        completion_language=MEM10_COMPLETION_LANGUAGE,
        forward_steps=forward_steps_map,
    )
    events_for_overlay = _events_for_overlay(n, [(s, pid) for s, pid in events])

    missing = [
        name
        for name, step in (
            ("tap", tap_step), ("grab", grab_spoon_step), ("scoop", scoop_step),
            ("pour", pour_step), ("place", place_step),
        )
        if step is None
    ]
    print(
        f"  events: tap={tap_step} grab={grab_spoon_step} scoop={scoop_step} "
        f"pour={pour_step} place={place_step} (n={n})"
        + (f"   MISSING: {', '.join(missing)}" if missing else "")
    )
    return labels, event_history, events_for_overlay


# Map task names to per-task labeling functions. Add more as we author them.
TASK_LABELERS = {
    "xarm7": label_xarm7,
    "xarm8": label_xarm8,
    "xarm9": label_xarm9,
    "xarm10": label_xarm10,
}


# ---------------------------------------------------------------------------
# Video overlay helpers
# ---------------------------------------------------------------------------

CAMERA_KEYS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")
DEFAULT_OVERLAY_CAMERA = "exterior_image_1_left"


def decode_frame(cell) -> np.ndarray:
    """Decode a LeRobot image cell to an RGB uint8 array.

    Handles the packed dict form ``{"bytes": ...}`` (and ``{"path": ...}``), raw
    PNG/JPEG bytes, and already-decoded arrays.
    """
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
        if cell.get("path"):
            return np.array(Image.open(cell["path"]).convert("RGB"))
        raise ValueError(f"image dict has neither bytes nor path: {list(cell)}")
    if isinstance(cell, (bytes, bytearray)):
        return np.array(Image.open(io.BytesIO(cell)).convert("RGB"))
    arr = np.asarray(cell)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def write_video(path: str, frames_bgr: list[np.ndarray], fps: float) -> None:
    """Write BGR frames as an H.264 (yuv420p) mp4 via imageio-ffmpeg.

    OpenCV's bundled FFmpeg has no H.264 encoder, so its ``mp4v`` output won't play
    in most browsers / players / VS Code. imageio's bundled ffmpeg has libx264 and
    yuv420p, which are broadly viewable.
    """
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    try:
        for f in frames_bgr:
            writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()


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
    grip_value: float,
    history_max_chars: int,
    event_labels: dict[int, str] | None = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    display_event = (event_labels or EVENT_LABELS).get(current_event, "(none)")
    hist_lines = wrap_text(event_history_text, max_chars=history_max_chars)

    small_h = 16
    big_h = 22
    pad = 8
    n_hist = max(1, len(hist_lines))
    block_h = pad + big_h + n_hist * small_h + small_h + pad

    y0 = h - block_h
    cv2.rectangle(frame, (5, y0), (w - 5, h - 5), (0, 0, 0), -1)

    y = y0 + pad + 16
    cv2.putText(frame, f"Event: {display_event}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y += big_h

    for hl in hist_lines:
        cv2.putText(frame, hl, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        y += small_h

    cv2.putText(frame, f"Event target: {target_label}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"grip:{grip_value:+.3f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return frame


# ---------------------------------------------------------------------------
# LeRobot metadata update (format-agnostic; identical to label_dataset_libero.py)
# ---------------------------------------------------------------------------


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

    stats_path = repo_root / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        print("  episodes_stats.jsonl not found, skipping stats update")
        return

    with open(stats_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

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
            vals = np.stack(df[col].values).astype(np.float64)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_DATASET_ROOT = "~/.cache/huggingface/lerobot/<your-hf-username>/mem10/data/chunk-000"
DEFAULT_TASK_NAME = "xarm10"
DEFAULT_EPISODE_START = 0
DEFAULT_EPISODE_END_EXCLUSIVE = 104   # mem10 has 104 episodes
DEFAULT_VIDEO_PREFIX = "xarm_debug"
DEFAULT_OUTPUT_DIR = "./dataset_videos"
DEFAULT_FPS = 20.0                    # mem7 is 20 fps
DEFAULT_FRAME_WIDTH = 320            # images are 240x320
DEFAULT_FRAME_HEIGHT = 240
DEFAULT_HISTORY_MAX_CHARS = 500


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Generate event labels and debug videos for xArm hardware episodes.")
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
    parser.add_argument(
        "--overlay-camera",
        choices=CAMERA_KEYS,
        default=DEFAULT_OVERLAY_CAMERA,
        help="Which camera stream to render the overlay video from.",
    )
    parser.add_argument("--history-max-chars", type=int, default=DEFAULT_HISTORY_MAX_CHARS)
    parser.add_argument("--transition-back", type=int, default=TRANSITION_BACK_STEPS)
    parser.add_argument("--transition-forward", type=int, default=TRANSITION_FORWARD_STEPS)
    parser.add_argument("--history-delay", type=int, default=HISTORY_DELAY_STEPS)
    # mem10 only. The right tap window depends on how long the hand was actually held
    # on the cup, which varies BETWEEN datasets — a short-hold recording wants ~3 while a
    # long-hold one wants ~30. Overriding here lets each dataset be labeled correctly
    # without editing the module constants between runs.
    parser.add_argument("--tap-back", type=int, default=None,
                        help="mem10: frames labeled before the tap marker "
                             f"(default {MEM10_TAP_BACK})")
    parser.add_argument("--tap-forward", type=int, default=None,
                        help="mem10: frames labeled after the tap marker. Set this to "
                             "cover the span the hand is genuinely on the cup — it also "
                             "bounds how many distinct keyframes the sampler can draw "
                             f"(default {MEM10_TAP_FORWARD})")
    parser.add_argument("--z-zero-tol", type=float, default=Z_ACTION_ZERO_TOLERANCE)
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write labels / phase_history back to parquet, then patch meta/info.json "
        "& episodes_stats.jsonl (unless --no-lerobot-meta-update).",
    )
    parser.add_argument(
        "--lerobot-repo-root",
        type=str,
        default="",
        help="LeRobot dataset root (folder containing meta/ and data/). If empty, inferred from --dataset-root.",
    )
    parser.add_argument(
        "--no-lerobot-meta-update",
        action="store_true",
        help="With --write-parquet, skip updating meta/info.json and meta/episodes_stats.jsonl.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()

    if args.task_name not in TASK_LABELERS:
        raise ValueError(
            f"No labeling function registered for task '{args.task_name}'. "
            f"Available: {sorted(TASK_LABELERS)}"
        )
    labeler = TASK_LABELERS[args.task_name]
    task_event_labels = TASK_EVENT_LABELS.get(args.task_name, EVENT_LABELS)

    # label_xarm10 reads these at call time, so overriding them here applies to this
    # run only and leaves the file defaults alone.
    global MEM10_TAP_BACK, MEM10_TAP_FORWARD  # noqa: PLW0603
    if args.tap_back is not None:
        MEM10_TAP_BACK = args.tap_back
    if args.tap_forward is not None:
        MEM10_TAP_FORWARD = args.tap_forward
    if args.task_name == "xarm10_v4":
        print(f"Tap label window: -{MEM10_TAP_BACK} .. +{MEM10_TAP_FORWARD} "
              f"({MEM10_TAP_BACK + MEM10_TAP_FORWARD + 1} frames = "
              f"{(MEM10_TAP_BACK + MEM10_TAP_FORWARD + 1) / args.fps:.2f}s)")

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

        # Pass episode_num to labeler if it supports it (for xarm10)
        labeler_kwargs = {
            "transition_back": args.transition_back,
            "transition_forward": args.transition_forward,
            "history_delay": args.history_delay,
            "z_zero_tol": args.z_zero_tol,
        }
        if args.task_name == "xarm10":
            labeler_kwargs["episode_num"] = episode_num

        labels, event_history_values, events_for_overlay = labeler(df, **labeler_kwargs)

        if args.write_parquet:
            df["labels"] = np.asarray(labels, dtype=np.int64)
            df["phase_history"] = event_history_values
            df.to_parquet(parquet_path, index=False)
            wrote_parquet_any = True
            print(f"  wrote labels, phase_history -> {parquet_path}")
        else:
            print("  dry-run (use --write-parquet to persist)")

        grip = gripper(df)
        frames: list[np.ndarray] = []
        for i in range(len(df)):
            row = df.iloc[i]
            rgb = decode_frame(row[args.overlay_camera])
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frame = draw_overlay(
                frame,
                current_event=events_for_overlay[i],
                event_history_text=event_history_values[i],
                target_label=labels[i],
                grip_value=float(grip[i]),
                history_max_chars=args.history_max_chars,
                event_labels=task_event_labels,
            )
            frames.append(frame)

        video_name = f"{args.video_prefix}_{episode_num:06d}.mp4"
        video_path = output_dir / video_name
        write_video(str(video_path), frames, args.fps)
        print(f"  saved {video_path}")

    if args.write_parquet and wrote_parquet_any and not args.no_lerobot_meta_update:
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
