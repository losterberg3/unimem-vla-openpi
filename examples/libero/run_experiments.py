#!/usr/bin/env python3
"""Run a batch of LIBERO experiments defined in a YAML file.

Manages the serve_policy server lifecycle (start / wait-ready / stop), calls
run_event_inference for each experiment, and scores the resulting event
histories using the ``scoring:`` block from the YAML.

Writes one output file next to the input YAML, named after its filename stem:
    <stem>_episodes.txt   Per-episode scoring breakdown for every experiment,
                          appended as each one completes.
A per-experiment summary table (success/failure counts, mean score, mean
rollout time, mean inference latency, etc.) is printed to the console after
the last experiment finishes.

Usage:
    uv run examples/libero/run_experiments.py experiments.yaml [--dry-run]
"""

import dataclasses
import logging
import pathlib
import subprocess
import sys
import time

import numpy as np
import yaml

# Make libero_inference and score_rollouts importable without installing.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from libero_inference import Args, run_event_inference
from score_rollouts import score_rollouts

_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SERVER_READY_TIMEOUT = 300  # seconds to wait for serve_policy to accept connections
_SERVER_POLL_INTERVAL = 3    # seconds between readiness probes


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------

def _server_key(server_cfg: dict) -> str:
    """Unique string identifying a (config, dir, port) triple."""
    return f"{server_cfg.get('config', '')}::{server_cfg.get('dir', '')}::{server_cfg.get('port', 8000)}"


def _start_server(server_cfg: dict) -> subprocess.Popen:
    port = server_cfg.get("port", 8000)
    cmd = [
        "uv", "run", "python", "scripts/serve_policy.py",
        "--port", str(port),
        "policy:checkpoint",
        "--policy.config", server_cfg["config"],
        "--policy.dir", server_cfg["dir"],
    ]
    logging.info("Starting server: %s", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(_OPENPI_ROOT))


def _wait_for_server(host: str, port: int) -> None:
    import http.client
    deadline = time.monotonic() + _SERVER_READY_TIMEOUT
    logging.info("Waiting for server at %s:%d ...", host, port)
    while time.monotonic() < deadline:
        conn = http.client.HTTPConnection(host, port, timeout=2.0)
        try:
            conn.request("GET", "/healthz")
            resp = conn.getresponse()
            if resp.status == 200:
                logging.info("Server ready at %s:%d", host, port)
                return
        except ConnectionRefusedError:
            pass
        except Exception:
            pass
        finally:
            conn.close()
        time.sleep(_SERVER_POLL_INTERVAL)
    raise TimeoutError(f"Server {host}:{port} did not start within {_SERVER_READY_TIMEOUT}s")


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logging.info("Server stopped (exit=%s)", proc.returncode)


# ---------------------------------------------------------------------------
# Args + scoring config construction
# ---------------------------------------------------------------------------

_INFERENCE_FIELDS = {f.name for f in dataclasses.fields(Args)}
_NON_INFERENCE_KEYS = {"name", "server", "scoring"}


def _build_args(defaults: dict, experiment: dict) -> Args:
    """Merge defaults + experiment overrides into an Args instance."""
    merged = {**defaults}
    for k, v in experiment.items():
        if k not in _NON_INFERENCE_KEYS:
            merged[k] = v
    filtered = {k: v for k, v in merged.items() if k in _INFERENCE_FIELDS}
    args = Args()
    for k, v in filtered.items():
        setattr(args, k, v)
    return args


def _build_scoring_cfg(defaults: dict, experiment: dict) -> dict:
    """Merge scoring config: experiment-level overrides defaults-level.

    Also pulls ``vlm_sequence`` from the experiment/defaults into the scoring
    config (if not already set there) so scorers can enforce exact event counts.
    """
    default_scoring = defaults.get("scoring", {})
    exp_scoring = experiment.get("scoring", {})
    merged = {**default_scoring, **exp_scoring}
    if "vlm_sequence" in experiment and "vlm_sequence" not in merged:
        merged["vlm_sequence"] = experiment["vlm_sequence"]
    return merged


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main(yaml_path: str, dry_run: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    yaml_path = str(pathlib.Path(yaml_path).expanduser().resolve())
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    defaults: dict = cfg.get("defaults", {})
    experiments: list[dict] = cfg.get("experiments", [])

    if not experiments:
        logging.error("No experiments defined in %s", yaml_path)
        sys.exit(1)

    default_host = defaults.get("host", "127.0.0.1")
    default_port = defaults.get("port", 8000)

    results: list[dict] = []
    current_server_key: str | None = None
    server_proc: subprocess.Popen | None = None
    episodes_txt_path = pathlib.Path(yaml_path).parent / f"{pathlib.Path(yaml_path).stem}_episodes.txt"
    if not dry_run:
        episodes_txt_path.write_text("")

    try:
        for exp in experiments:
            exp_name = exp.get("name", "unnamed")
            server_cfg: dict | None = exp.get("server", defaults.get("server"))
            scoring_cfg = _build_scoring_cfg(defaults, exp)

            # Determine server key for this experiment.
            if server_cfg is None:
                new_key = "__external__"
            else:
                new_key = _server_key({**{"port": default_port}, **server_cfg})

            # (Re)start server only when the checkpoint changes.
            if new_key != current_server_key:
                if server_proc is not None:
                    logging.info("Stopping previous server (%s)", current_server_key)
                    _stop_server(server_proc)
                    server_proc = None
                    time.sleep(2)

                if server_cfg is not None:
                    if dry_run:
                        logging.info("[dry-run] Would start server: config=%s dir=%s",
                                     server_cfg.get("config"), server_cfg.get("dir"))
                    else:
                        server_proc = _start_server(server_cfg)
                        host = server_cfg.get("host", default_host)
                        port = server_cfg.get("port", default_port)
                        _wait_for_server(host, port)
                else:
                    logging.info("Using externally managed server at %s:%d", default_host, default_port)

                current_server_key = new_key

            logging.info("=" * 70)
            logging.info("Experiment: %s", exp_name)
            logging.info("=" * 70)

            args = _build_args(defaults, exp)
            if server_cfg and "port" in server_cfg:
                args.port = server_cfg["port"]
            if not args.batch_name:
                args.batch_name = pathlib.Path(yaml_path).stem

            if dry_run:
                logging.info("[dry-run] Would run: %s  scoring: %s", args, scoring_cfg)
                rollouts: list[dict] = []
            else:
                exp_results = run_event_inference(args)
                rollouts = exp_results.get("rollouts", [])

            scored = score_rollouts(rollouts, scoring_cfg)
            scores = scored["scores"]
            dists  = scored["dists"]
            fails  = scored["failure_count"]

            # Per-episode breakdown (same lines score_rollouts logs to the console),
            # appended to a file after this experiment finishes.
            episode_lines = scored.get("episode_lines", [])
            if episode_lines and not dry_run:
                with open(episodes_txt_path, "a") as f:
                    f.write(f"{'=' * 70}\n")
                    f.write(f"Experiment: {exp_name}\n")
                    f.write(f"{'=' * 70}\n")
                    f.write("\n".join(episode_lines) + "\n\n")
                logging.info("Per-episode breakdown saved -> %s", episodes_txt_path)

            mean_time = scored.get("mean_total_time_s")
            mean_infer = scored.get("mean_infer_ms")

            results.append({
                "experiment":       exp_name,
                "num_rollouts":     args.num_rollouts,
                "successes":        len(scores),
                "failures":         fails,
                "mean_score":       f"{np.mean(scores):.4f}" if scores else "N/A",
                "std_score":        f"{np.std(scores):.4f}"  if scores else "N/A",
                "mean_dist_m":      f"{np.mean(dists):.4f}"  if dists  else "N/A",
                "mean_time_s":      f"{mean_time:.2f}" if mean_time is not None else "N/A",
                "mean_infer_ms":    f"{mean_infer:.1f}" if mean_infer is not None else "N/A",
                "mode":             args.mode,
                "scoring_type":     scoring_cfg.get("type", "none"),
                "checkpoint_dir":   server_cfg.get("dir", "external") if server_cfg else "external",
                "event_breakdown":  scored.get("event_breakdown", ""),
            })
            logging.info("Completed %d/%d experiments", len(results), len(experiments))

    finally:
        if server_proc is not None:
            _stop_server(server_proc)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    if not results:
        return

    col_order = [
        "experiment", "num_rollouts", "successes", "failures",
        "mean_score", "std_score", "mean_dist_m", "mean_time_s", "mean_infer_ms",
        "mode", "scoring_type",
    ]

    logging.info("")
    logging.info("=" * 70)
    logging.info("RESULTS SUMMARY")
    logging.info("=" * 70)

    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in results)) for c in col_order}
    header = "  ".join(c.ljust(widths[c]) for c in col_order)
    logging.info(header)
    logging.info("-" * len(header))
    for row in results:
        logging.info("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in col_order))
    logging.info("")


if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    if not args:
        print(f"Usage: {sys.argv[0]} <experiments.yaml> [--dry-run]")
        sys.exit(1)
    main(args[0], dry_run=dry)
