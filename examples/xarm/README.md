# xArm Real-Robot Memory Experiments

This is the real-robot counterpart to [`examples/libero`](../libero): memory-conditioned
policy rollouts (event history text, keyframes, and/or a video encoder) on a physical
xArm, instead of in simulation.

## Hardware this was built on

This code was written and tested against one specific physical rig:

- **Robot**: one [UFactory xArm](https://www.ufactory.cc/xarm/), driven via `xarm-python-sdk`
  in servo-cartesian mode (`arm.set_servo_cartesian`) at 40 Hz, with a 6-DoF end-effector
  pose (xyz + rpy) plus a 1-DoF gripper. Its home pose and IP address are specific to our
  unit.
- **Cameras**: two Intel RealSense cameras (one exterior-mounted, one wrist-mounted),
  read via `pyrealsense2` at 320x240@30fps and identified by USB serial number.
- **Gripper**: the standard UFactory xArm gripper, whose 0–850 tick range is baked into
  the tick-conversion arithmetic in `get_observation()` / the execution loop in
  `xarm_inference.py` and in `demo_collection.py`. That range is a property of this
  gripper model, not a per-lab config value, so it isn't exposed as a flag.

Both dependencies (`xarm-python-sdk`, `pyrealsense2`) are already declared in the
top-level `pyproject.toml`, so a normal `uv sync` from the repo root is enough — there's
no separate venv/Docker setup for this example (unlike `examples/libero`, which needs
one for LIBERO's own dependency constraints).

## Adapting to your own hardware

Everything above is almost certainly different on your setup. Two tiers of change:

**No code changes needed** — the arm's IP and both camera serials are CLI-overridable
in both `xarm_inference.py` (`--arm-ip`, `--camera-serial-external`,
`--camera-serial-wrist`, plus `--home-x/y/z/roll/pitch/yaw` for the home pose and
`--collision-sensitivity` for the arm's collision-detection sensitivity (0=off ..
5=very high) — these are all `Args` fields, so they also work as keys in an
experiment yaml under `experiments/`) and `demo_collection.py` (same three
`--arm-ip` / `--camera-serial-external` / `--camera-serial-wrist` flags; it has no
home pose or collision-sensitivity flag since it never calls `_go_home()`). Find
your RealSense serials with:

```bash
rs-enumerate-devices | grep Serial
```

**Code changes needed** — if you're not using a UFactory xArm and/or Intel RealSense
cameras, you'll need to rewrite the functions that actually talk to the hardware SDK:

- `_go_home()`, `interpolate_action()`, and the gripper tick math in `get_observation()`
  and the execution loop in `xarm_inference.py` all call directly into `XArmAPI`. Swap
  these for your own arm's control API. The action convention downstream of these
  (6-DoF cartesian pose + normalized gripper command) is specific to this manipulation
  setup — a different DoF count or control mode (e.g. joint-space) needs a matching
  change in the policy's action head, not just this script.
- `_start_cameras()` and the camera reads in `get_observation()` call directly into
  `pyrealsense2`. Swap in your own camera SDK's frame-grab calls — the only requirement
  downstream is that each frame ends up as an HxWx3 uint8 RGB array under the same obs
  keys (`observation/exterior_image_1_left`, `observation/wrist_image_left`).

Everything else in `xarm_inference.py` — event detection, event-history text, keyframe
memory, the video encoder's frame buffering, the threaded inference/execution loop — is
hardware-agnostic and should carry over to a different arm/camera unchanged.

`demo_collection.py` (teleoperated demo recording) has the same hardware assumptions.
Arm IP and camera serials are CLI flags (see above); the dataset repo name and task
description change per recording session rather than per rig, so those stay as plain
constants in the `# Task config` block at the top of the file — edit them directly
before running.

## Running inference

Serve a checkpoint (see the main [README](../../README.md) for training your own):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config <config-name> --policy.dir <checkpoint-dir>
```

Then either run a single rollout directly:

```bash
uv run examples/xarm/xarm_inference.py --host <server-host> --task mem7 \
    --arm-ip <your-arm-ip> --camera-serial-external <serial> --camera-serial-wrist <serial>
```

or run a batch of experiments (one policy server per checkpoint, launched/torn down
automatically) from a yaml, mirroring `examples/libero/run_experiments.py`:

```bash
uv run examples/xarm/run_experiments.py examples/xarm/experiments/mem7.yaml
uv run examples/xarm/run_experiments.py examples/xarm/experiments/mem7.yaml --dry-run
```

`experiments/mem7.yaml` .. `mem10.yaml` are the actual batch configs used for the memory
ablations in the paper (one file per task; each runs several memory-mode arms against
their respective checkpoints). Unlike `examples/libero/run_experiments.py`, there's no
scoring step or CSV output here — hardware rollouts aren't auto-scorable the way
simulated ones are — just a console latency summary per experiment.

## Collecting your own demos

`demo_collection.py` records teleoperated episodes straight to a LeRobot dataset. It's
driven by an external process touching flag files in `/tmp`
(`/tmp/start_demo`, `/tmp/stop_demo`, `/tmp/cup_tap`) rather than by keyboard input in the
same process, so it can run alongside whatever teleop controller you use.

```bash
uv run examples/xarm/demo_collection.py \
    --arm-ip <your-arm-ip> --camera-serial-external <serial> --camera-serial-wrist <serial>
```

## Labeling collected data

`label_dataset_xarm.py` reads a recorded LeRobot dataset and generates per-timestep
event labels + event-history text (the supervision the memory-conditioned model trains
on). Unlike the two scripts above, it never talks to the arm or cameras directly — it
only reads already-recorded episodes — so there's no arm IP/camera serial to configure;
`--dataset-root`, `--frame-width`/`--frame-height`, and `--fps` (already CLI flags) are
the only rig-shaped values it depends on, and they only need to match whatever produced
the dataset you're labeling. See its module docstring for the class scheme and per-task
labeler structure.
