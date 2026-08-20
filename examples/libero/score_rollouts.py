"""Score LIBERO rollouts from event history, using criteria defined in the experiment YAML.

Scoring config lives under a ``scoring:`` key in the YAML and is passed here as a plain dict.
``run_experiments.py`` collects rollout dicts from ``run_event_inference``, calls
``score_rollouts`` to produce summary metrics, prints a per-experiment summary table to
the console, and writes a per-episode text breakdown to ``<yaml-stem>_episodes.txt`` —
see the module docstring in run_experiments.py for the exact output files.

Supported scoring types
-----------------------
grab_place_xy
    Gaussian score on the XY distance between two named events.
    Uses EE positions recorded from the environment at event fire time.

    YAML keys:
        event_a  (str)   — chunk prefix for the source event, e.g. "grabbed box"
        event_b  (str)   — chunk prefix for the target event, e.g. "placed box"
        sigma    (float) — Gaussian kernel sigma in metres (default 0.08)

drop_tap_match
    Binary: 1.0 if the side the box was dropped into matches the side that was tapped.
    Succeeds whether the robot chose left or right, as long as drop and tap agree.
    Fails if: sides disagree, drop event is missing, or tap event is missing.

    YAML keys:
        drop_left  (str) — chunk prefix for dropping into the left basket, e.g. "dropped left"
        drop_right (str) — chunk prefix for dropping into the right basket, e.g. "dropped right"
        tap_left   (str) — chunk prefix for tapping the left basket, e.g. "tapped left basket"
        tap_right  (str) — chunk prefix for tapping the right basket, e.g. "tapped right basket"

object_return_xy
    Gaussian score on the XY displacement of a tracked object between the start and
    end of the rollout — e.g. did the robot put it back where it found it?
    Requires ``track_object_pos_key`` set on the inference Args, so the rollout dict
    carries ``initial_obj_xy``/``final_obj_xy``.

    YAML keys:
        sigma (float) — Gaussian kernel sigma in metres (default 0.05)

initial_obj_to_event_xy
    Gaussian score on the XY distance between an object's initial position and the EE
    position when a named event fired — e.g. did the robot act near where the object
    started? Requires ``track_object_pos_key`` set on the inference Args.

    YAML keys:
        event (str)   — chunk prefix for the event to compare against, e.g. "tapped plate"
        sigma (float) — Gaussian kernel sigma in metres (default 0.08)

event_completion
    Binary: 1.0 if event counts match exactly, else failure.

    When ``vlm_sequence`` is present in the scoring config (automatically pulled from the
    experiment block by run_experiments.py), the scorer counts how many times each event
    prefix appears in the completed chunks and requires an exact match. Extra or missing
    repetitions both count as failure.

    Falls back to the legacy behaviour (each event in ``required_events`` appears at least
    once) when no ``vlm_sequence`` is available.

    YAML keys:
        vlm_sequence    (list[str]) — the full ordered expected event sequence; counts are derived from this
        required_events (list[str]) — fallback: chunk prefixes that must each appear at least once

none
    No scoring; every rollout is counted in the total but not in successes.

Example YAML block
------------------
scoring:
  type: "grab_place_xy"
  event_a: "grabbed box"
  event_b: "placed box"
  sigma: 0.08
"""

import logging
import numpy as np

def _score_grab_place_xy(
    rollout: dict,
    event_a: str,
    event_b: str,
    sigma: float,
) -> tuple[float, float] | None:
    """Gaussian score on XY distance between two named events.

    Uses EE positions recorded from the environment at event fire time (rollout["event_xy"]).
    """
    event_xy = rollout.get("event_xy", {})
    pos_a = event_xy.get(event_a)
    pos_b = event_xy.get(event_b)
    if pos_a is None or pos_b is None:
        return None
    dist = float(np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2))
    score = float(np.exp(-dist ** 2 / (2 * sigma ** 2)))
    return score, dist


def _score_initial_obj_to_event_xy(
    rollout: dict,
    event: str,
    sigma: float,
) -> tuple[float, float] | None:
    """Gaussian score on XY distance between initial object position and EE at a named event.

    Uses rollout["initial_obj_xy"] (object position before the robot acts) and
    rollout["event_xy"][event] (EE position when the event fired).
    """
    initial = rollout.get("initial_obj_xy")
    event_xy = rollout.get("event_xy", {})
    ee_at_event = event_xy.get(event)
    if initial is None or ee_at_event is None:
        return None
    dist = float(np.sqrt((initial[0] - ee_at_event[0]) ** 2 + (initial[1] - ee_at_event[1]) ** 2))
    score = float(np.exp(-dist ** 2 / (2 * sigma ** 2)))
    return score, dist


def _score_drop_tap_match(chunks: list[str], scoring_cfg: dict) -> tuple[float, str] | None:
    """Return (1.0, side) if drop and tap sides match, else None."""
    drop_left  = scoring_cfg.get("drop_left",  "dropped left")
    drop_right = scoring_cfg.get("drop_right", "dropped right")
    tap_left   = scoring_cfg.get("tap_left",   "tapped left basket")
    tap_right  = scoring_cfg.get("tap_right",  "tapped right basket")

    dropped = next(
        ("left" if c.startswith(drop_left) else "right"
         for c in chunks if c.startswith(drop_left) or c.startswith(drop_right)),
        None,
    )
    tapped = next(
        ("left" if c.startswith(tap_left) else "right"
         for c in chunks if c.startswith(tap_left) or c.startswith(tap_right)),
        None,
    )

    if dropped is None or tapped is None:
        return None
    if dropped == tapped:
        return (1.0, dropped)
    return None  # mismatch


def _score_object_return_xy(rollout: dict, sigma: float) -> tuple[float, float] | None:
    """Gaussian score on XY displacement of a tracked object from start to end of rollout."""
    initial = rollout.get("initial_obj_xy")
    final = rollout.get("final_obj_xy")
    if initial is None or final is None:
        return None
    dist = float(np.sqrt((final[0] - initial[0]) ** 2 + (final[1] - initial[1]) ** 2))
    score = float(np.exp(-dist ** 2 / (2 * sigma ** 2)))
    return score, dist


def score_rollouts(rollouts: list[dict], scoring_cfg: dict) -> dict:
    """Score a batch of rollouts and return summary metrics.

    Each rollout dict must have:
        chunks        (list[str]) — completed event chunks, e.g. ["grabbed box 45 09 70", ...]
        event_history (str)       — formatted event history string (for logging)

    Returns a dict with:
        scores        (list[float])
        dists         (list[float])  — secondary metric (XY dist for grab_place_xy, else 0.0)
        failure_count (int)
        summary       (str)          — human-readable summary line
    """
    scoring_type = scoring_cfg.get("type", "none")
    scores: list[float] = []
    dists: list[float] = []
    failure_count = 0

    # Plain-text mirror of every logging.info(...) call below, so callers can dump a
    # per-episode breakdown to a file instead of only ever seeing it scroll by in the
    # terminal. Same content, same order, just captured instead of only logged.
    episode_lines: list[str] = []

    def _log(msg: str, *args) -> None:
        formatted = msg % args if args else msg
        logging.info(msg, *args)
        episode_lines.append(formatted)

    # Per-event hit counts (event_completion only): how many rollouts each named
    # event actually fired in, independent of overall binary success/failure.
    event_names: list[str] = []
    if scoring_type == "event_completion":
        vlm_sequence = scoring_cfg.get("vlm_sequence", [])
        event_names = list(dict.fromkeys(vlm_sequence)) if vlm_sequence else list(scoring_cfg.get("required_events", []))
    event_hits: dict[str, int] = {name: 0 for name in event_names}

    for i, rollout in enumerate(rollouts):
        chunks = rollout.get("chunks", [])
        result = None

        for name in event_names:
            if any(c.startswith(name) for c in chunks):
                event_hits[name] += 1

        if scoring_type == "drop_tap_match":
            match = _score_drop_tap_match(chunks, scoring_cfg)
            if match is not None:
                result = (match[0], 0.0)
                _log("  [rollout %d] drop_tap_match: success (side=%s)", i + 1, match[1])
            else:
                dropped_ev = next((c for c in chunks if c.startswith("dropped")), None)
                tapped_ev  = next((c for c in chunks if c.startswith("tapped")), None)
                _log(
                    "  [rollout %d] drop_tap_match: FAIL — dropped=%r tapped=%r",
                    i + 1, dropped_ev, tapped_ev,
                )

        elif scoring_type == "grab_place_xy":
            result = _score_grab_place_xy(
                rollout,
                event_a=scoring_cfg.get("event_a", "grabbed box"),
                event_b=scoring_cfg.get("event_b", "placed box"),
                sigma=float(scoring_cfg.get("sigma", 0.08)),
            )
            if result is not None:
                score, dist = result
                _log(
                    "  [rollout %d] score=%.4f  dist=%.4f m  (sigma=%.2f m)",
                    i + 1, score, dist, float(scoring_cfg.get("sigma", 0.08)),
                )

        elif scoring_type == "object_return_xy":
            result = _score_object_return_xy(rollout, sigma=float(scoring_cfg.get("sigma", 0.05)))
            if result is not None:
                score, dist = result
                _log(
                    "  [rollout %d] object_return_xy: score=%.4f  dist=%.4f m  (sigma=%.2f m)",
                    i + 1, score, dist, float(scoring_cfg.get("sigma", 0.05)),
                )
            else:
                _log("  [rollout %d] object_return_xy: FAIL — no object XY tracked", i + 1)

        elif scoring_type == "initial_obj_to_event_xy":
            event = scoring_cfg.get("event", "")
            result = _score_initial_obj_to_event_xy(rollout, event=event, sigma=float(scoring_cfg.get("sigma", 0.08)))
            if result is not None:
                score, dist = result
                _log(
                    "  [rollout %d] initial_obj_to_event_xy: score=%.4f  dist=%.4f m  event='%s'",
                    i + 1, score, dist, event,
                )
            else:
                _log(
                    "  [rollout %d] initial_obj_to_event_xy: FAIL — missing initial_obj_xy or event_xy['%s']",
                    i + 1, event,
                )

        elif scoring_type == "event_completion":
            vlm_sequence = scoring_cfg.get("vlm_sequence", [])
            if vlm_sequence:
                # Build expected counts from vlm_sequence (exact repetitions required).
                expected: dict[str, int] = {}
                for ev in vlm_sequence:
                    expected[ev] = expected.get(ev, 0) + 1
                actual: dict[str, int] = {
                    ev: sum(1 for c in chunks if c.startswith(ev)) for ev in expected
                }
                if actual == expected:
                    result = (1.0, 0.0)
                    _log("  [rollout %d] event_completion: success (counts=%s)", i + 1, actual)
                else:
                    _log(
                        "  [rollout %d] event_completion: FAIL — expected %s got %s",
                        i + 1, expected, actual,
                    )
            else:
                required = scoring_cfg.get("required_events", [])
                if all(any(c.startswith(ev) for c in chunks) for ev in required):
                    result = (1.0, 0.0)
                    _log("  [rollout %d] event_completion: success", i + 1)

        if result is None:
            failure_count += 1
            _log(
                "  [rollout %d] SCORE MISS — event_history: %s",
                i + 1, rollout.get("event_history", "none"),
            )
        else:
            scores.append(result[0])
            dists.append(result[1])

    n = len(rollouts)
    if n > 0 and scoring_type != "none":
        _log("=" * 60)
        _log("SCORING SUMMARY  type=%s  rollouts=%d", scoring_type, n)
        _log("  Successes: %d  Failures: %d", len(scores), failure_count)
        if scores:
            _log("  Mean score: %.4f  Std: %.4f", float(np.mean(scores)), float(np.std(scores)))
            if scoring_type == "grab_place_xy" and dists:
                _log("  Mean dist:  %.4f m", float(np.mean(dists)))
        if event_names:
            _log("  Event breakdown:")
            for name in event_names:
                _log("    %-20s %d/%d", name, event_hits[name], n)
        _log("=" * 60)

    event_breakdown_str = (
        " | ".join(f"{name}={event_hits[name]}/{n}" for name in event_names) if event_names else ""
    )

    summary = (
        f"successes={len(scores)}/{n}  mean_score={np.mean(scores):.4f}" if scores else f"successes=0/{n}"
    )

    times_s = [r["total_time_s"] for r in rollouts if "total_time_s" in r]
    infer_ms = [r["avg_infer_ms"] for r in rollouts if "avg_infer_ms" in r]

    return {
        "scores": scores,
        "dists": dists,
        "failure_count": failure_count,
        "event_breakdown": event_breakdown_str,
        "summary": summary,
        "mean_total_time_s": round(float(np.mean(times_s)), 2) if times_s else None,
        "mean_infer_ms": round(float(np.mean(infer_ms)), 1) if infer_ms else None,
        "episode_lines": episode_lines,
    }
