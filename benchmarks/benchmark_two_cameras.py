#!/usr/bin/env python3
"""Latency vs. context length for the real xarm rig's 2 camera streams
(base_0_rgb, left_wrist_0_rgb — xarm_policy.py:60-63): three π0.5
comparisons — Base Single-Frame, Naive Video, Ours (Keyframe Caching).

Three modes, one Pi0Config per point (video_encoder=True throughout —
num_frames=1 must still go through the VideoEncoder family, not
video_encoder=False, or "base" measures an unrelated scan-vs-unroll slowdown;
see pi0.py: scan=(not video_encoder)):

  base   (K=1)        - a single current frame, no history. Also the K=1 point
                         for "naive" and "cached" below (with zero history all
                         three collapse to the same computation).
  naive  (K=2/4/8/16) - client stacks the full K-frame history itself every
                         call; the server never gets a reset_cache signal, so
                         it re-encodes all K frames through the full
                         VideoEncoder each time (xarm_inference.py's "video"
                         mode: np.stack(hist + [raw], axis=0)).
  cached (K=1..16)     - client sends only the current frame; the first call
                         signals reset_cache (Policy seeds each camera's cache
                         with K-1 zero-history frames on first appearance —
                         policy.py:234-249), every later call omits it, so the
                         cache is seeded once and never slid again — same as
                         xarm_inference.py's "no_memory"/pinned keyframe modes.

Each point spins up a real WebsocketPolicyServer wrapping a real Policy, and a
real WebsocketClientPolicy talking to it over localhost — the actual serving
stack, not a bare JAX function call — in its own subprocess, so its model and
websocket server are fully torn down before the next point starts (otherwise
the previous subprocess's model is still resident on the GPU when the next
one is built). The reported latency is the `server_timing.infer_ms` field the
server returns with every response: a purely server-side measurement
(websocket_policy_server.py's handler times the whole `Policy.infer()` call
with time.monotonic()) that has no client-side scheduling jitter, and — unlike
Policy's own internal `policy_timing.infer_ms`, which only wraps
sample_actions_event — still counts the SigLIP cache-encode step where applicable.

Model hyperparameters are hardcoded to match the "xarm_mem8_infer" TrainConfig
(src/openpi/training/config.py:1428) — pi05, action_dim=32, action_horizon=50,
gemma_2b_lora + gemma_300m_lora, event_tracking=True. K=4 uses the num_frames
xarm_mem8_infer was actually trained with; other K are the same architecture
at hypothetical other context lengths (valid for latency, which depends on
tensor shapes, not trained weights — random-init params are fine).

Prints median/std latency (ms) per point; redirect stdout or edit the arrays
in your own plotting script to visualize the results.

Usage:
    uv run benchmarks/benchmark_two_cameras.py
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

_CACHE_DIR = os.path.expanduser("~/.cache/openpi_xla")
_CAMERAS = ["base_0_rgb", "left_wrist_0_rgb"]  # PI0/PI05 has no right_wrist_0_rgb slot
_H, _W = 224, 224
_N_WARMUP = 10
_N_TIMED = 500
_BASE_PORT = 18760

_KEYFRAMES_CACHED = [1, 2, 4, 8, 16]
_KEYFRAMES_NAIVE = [2, 4, 8, 16]  # K=1 reuses the base measurement (no history either way)


def _run_worker(spec: dict) -> dict:
    """Runs inside an isolated subprocess for exactly one (mode, K) point."""
    import threading

    import jax

    from openpi.models.pi0_config import Pi0Config
    from openpi.policies import policy as _policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    os.makedirs(_CACHE_DIR, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

    mode = spec["mode"]  # "base" | "naive" | "cached"
    k = spec["num_frames"]
    cfg = Pi0Config(
        # Matches xarm_mem8_infer (config.py:1428) exactly, except
        # video_encoder/num_frames, which this sweep varies.
        pi05=True,
        action_dim=32,
        action_horizon=50,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        event_tracking=True,
        video_encoder=True,
        num_frames=k,
    )
    rng = jax.random.key(0)
    model = cfg.create(rng)
    # transforms=[]: send raw Observation.from_dict-format dicts directly
    # (image/image_mask/state/tokenized_prompt/tokenized_prompt_mask), skipping
    # the real XarmInputs/Normalize pipeline (needs norm_stats from a real
    # checkpoint) — that pipeline is cheap numpy/CPU work, not the GPU-bound
    # cost this benchmark cares about.
    policy = _policy.Policy(model, rng=rng, transforms=[], output_transforms=[])

    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=spec["port"])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.2)  # give the async server a moment to bind before connecting
    client = WebsocketClientPolicy(host="127.0.0.1", port=spec["port"])

    if mode == "cached":
        # Only "cached" sends a single un-stacked frame — its history lives
        # server-side in the SigLIP hidden-state cache (raw pre-temporal-PE,
        # pre-projection activations per temporal layer, not a standard
        # transformer KV cache — see siglip_hidden_cache.py), and
        # Policy.infer() injects pre_encoded_images for it, which embed_prefix
        # (pi0.py) reads through a branch that never touches obs.image_masks
        # at all.
        img = np.zeros((_H, _W, 3), dtype=np.float32)
        mask = np.ones((), dtype=bool)
    else:
        # "naive" (any K) and "base" (K=1) both go through embed_prefix's
        # plain/no-cache branch instead, which — whenever video_encoder=True —
        # indexes obs.image_masks[name][:, -1] unconditionally (pi0.py:161),
        # so it needs a real temporal axis even at K=1, not a scalar mask.
        img = np.zeros((k, _H, _W, 3), dtype=np.float32)
        mask = np.ones((k,), dtype=bool)
    obs = {
        "image": {c: img for c in _CAMERAS},
        "image_mask": {c: mask for c in _CAMERAS},
        "state": np.zeros((cfg.action_dim,), dtype=np.float32),
        "tokenized_prompt": np.zeros((cfg.max_token_len,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((cfg.max_token_len,), dtype=bool),
    }

    if mode == "cached":
        # First call signals reset_cache (seeds K-1 zero-history frames) and
        # pays the XLA compile — same as xarm_inference.py's
        # init_obs["reset_cache"]=True. Discard from steady-state timing.
        client.infer({**obs, "reset_cache": True})
    else:
        client.infer(obs)  # first call: pays the XLA compile only

    for _ in range(_N_WARMUP):
        client.infer(obs)

    times = []
    for _ in range(spec["timed"]):
        out = client.infer(obs)
        times.append(out["server_timing"]["infer_ms"])

    arr = np.array(times)
    return {
        "mode": mode,
        "k": k,
        "median_ms": round(float(np.median(arr)), 1),
        "std_ms": round(float(arr.std()), 1),
    }


def _run_point(mode: str, k: int, port: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name
    spec = {"mode": mode, "num_frames": k, "port": port, "timed": _N_TIMED, "result_path": result_path}
    proc = subprocess.run([sys.executable, __file__, "--_worker_spec", json.dumps(spec)], check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Worker for mode={mode!r} K={k} failed (exit {proc.returncode}) — see output above.")
    result = json.loads(pathlib.Path(result_path).read_text())
    pathlib.Path(result_path).unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--_worker_spec", help=argparse.SUPPRESS)  # internal: run one point, isolated
    args = parser.parse_args()

    if args._worker_spec:
        spec = json.loads(args._worker_spec)
        result = _run_worker(spec)
        pathlib.Path(spec["result_path"]).write_text(json.dumps(result))
        return

    print(f"Sweeping latency across {len(_CAMERAS)} camera streams "
          "(each point in its own subprocess: clean GPU memory + a fresh websocket server)...\n")

    port = _BASE_PORT

    print(f"{'=' * 60}\n  base (K=1)\n{'=' * 60}")
    base = _run_point("base", 1, port)
    port += 1
    print(f"  median={base['median_ms']:.1f}ms  std={base['std_ms']:.1f}ms\n")

    print(f"{'=' * 60}\n  naive video\n{'=' * 60}")
    naive = {1: base}
    for k in _KEYFRAMES_NAIVE:
        naive[k] = _run_point("naive", k, port)
        port += 1
        print(f"  K={k:2d}  median={naive[k]['median_ms']:6.1f}ms  std={naive[k]['std_ms']:5.1f}ms")

    print(f"\n{'=' * 60}\n  keyframe-cached\n{'=' * 60}")
    cached = {}
    for k in _KEYFRAMES_CACHED:
        cached[k] = _run_point("cached", k, port)
        port += 1
        print(f"  K={k:2d}  median={cached[k]['median_ms']:6.1f}ms  std={cached[k]['std_ms']:5.1f}ms")

    keyframes = np.array(_KEYFRAMES_CACHED)
    latency_pi05_base = np.full(len(keyframes), base["median_ms"])
    latency_naive_video = np.array([naive[k]["median_ms"] for k in keyframes])
    latency_ours_cached = np.array([cached[k]["median_ms"] for k in keyframes])

    print(f"\n{'=' * 70}\nRESULTS ({len(_CAMERAS)} camera streams)\n{'=' * 70}")
    print("keyframes           =", list(keyframes))
    print("latency_pi05_base   =", list(latency_pi05_base))
    print("latency_naive_video =", list(latency_naive_video))
    print("latency_ours_cached =", list(latency_ours_cached))


if __name__ == "__main__":
    main()
