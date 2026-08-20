#!/usr/bin/env python3
"""Run a batch of xArm hardware experiments defined in a YAML file.

Analogous to examples/libero/run_experiments.py but for xArm hardware, and
simpler: there's no scoring step (hardware rollouts aren't auto-scorable the
way simulated ones are — see score_rollouts.py) and no CSV output, just a
per-experiment inference-latency summary printed to the console at the end.
Manages the serve_policy server lifecycle (start / wait-ready / stop) and
calls run_event_inference for each experiment.

Usage:
    uv run examples/xarm/run_experiments.py examples/xarm/experiments/mem7.yaml
    uv run examples/xarm/run_experiments.py examples/xarm/experiments/mem7.yaml --dry-run

Hardware fields (arm IP, camera serials, home pose) can be overridden per-yaml the
same way as any other xarm_inference.Args field — see examples/xarm/README.md.
"""

import dataclasses
import logging
import os
import pathlib
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from xarm_inference import Args, run_event_inference

_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SERVER_READY_TIMEOUT = 300  # seconds to wait for serve_policy to accept connections
_SERVER_POLL_INTERVAL = 3    # seconds between readiness probes


# ---------------------------------------------------------------------------
# Server lifecycle (mirrors libero/run_experiments.py)
# ---------------------------------------------------------------------------

def _server_key(server_cfg: dict) -> str:
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
    return subprocess.Popen(cmd, cwd=str(_OPENPI_ROOT), start_new_session=True)


def _wait_for_server(host: str, port: int) -> None:
    import http.client
    deadline = time.monotonic() + _SERVER_READY_TIMEOUT
    logging.info("Waiting for server at %s:%d ...", host, port)
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2.0)
            conn.request("GET", "/healthz")
            resp = conn.getresponse()
            if resp.status == 200:
                logging.info("Server ready at %s:%d", host, port)
                return
        except Exception:
            pass
        finally:
            conn.close()
        time.sleep(_SERVER_POLL_INTERVAL)
    raise TimeoutError(f"Server {host}:{port} did not start within {_SERVER_READY_TIMEOUT}s")


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    logging.info("Server stopped")


# ---------------------------------------------------------------------------
# Args construction
# ---------------------------------------------------------------------------

_INFERENCE_FIELDS = {f.name for f in dataclasses.fields(Args)}
_NON_INFERENCE_KEYS = {"name", "server"}


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


# ---------------------------------------------------------------------------
# Main
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

    try:
        for exp in experiments:
            exp_name = exp.get("name", "unnamed")
            server_cfg: dict | None = exp.get("server")

            new_key = _server_key({**{"port": default_port}, **server_cfg}) if server_cfg else "__external__"

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

            if dry_run:
                logging.info("[dry-run] Would run: %s", args)
                rollouts: list[dict] = []
            else:
                exp_results = run_event_inference(args)
                rollouts = exp_results.get("rollouts", [])

            infer_means = [r["mean_infer_s"] for r in rollouts if "mean_infer_s" in r]
            infer_mins  = [r["min_infer_s"]  for r in rollouts if "min_infer_s"  in r]
            infer_maxs  = [r["max_infer_s"]  for r in rollouts if "max_infer_s"  in r]

            results.append({
                "experiment":     exp_name,
                "num_rollouts":   args.num_rollouts,
                "mode":           args.mode,
                "mean_infer_s":   f"{np.mean(infer_means):.4f}" if infer_means else "N/A",
                "min_infer_s":    f"{np.min(infer_mins):.4f}"   if infer_mins  else "N/A",
                "max_infer_s":    f"{np.max(infer_maxs):.4f}"   if infer_maxs  else "N/A",
                "checkpoint_dir": server_cfg.get("dir", "external") if server_cfg else "external",
            })
            logging.info("Completed %d/%d experiments", len(results), len(experiments))

    except KeyboardInterrupt:
        logging.info("Interrupted — killing server")
    finally:
        if server_proc is not None:
            _stop_server(server_proc)

    if not results:
        return

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    col_order = [
        "experiment", "num_rollouts", "mode",
        "mean_infer_s", "min_infer_s", "max_infer_s",
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
