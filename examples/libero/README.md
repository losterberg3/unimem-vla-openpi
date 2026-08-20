# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

## Memory-conditioned experiments (this fork)

`main.py` above is upstream openpi's standard, memory-free LIBERO eval loop. This fork
adds a separate, parallel workflow for the memory-conditioned tasks used in the paper
(event history text, keyframes, and/or a video encoder) — see the [root README's
terminology note](../../README.md#a-note-on-terminology) for the "event" vs. "phase"
naming.

- **`libero_inference.py`** — the memory-aware rollout client (`Args`,
  `run_event_inference`). Analogous to `main.py` but supports the `no_memory` / `text` /
  `text_keyframe` / `keyframe` / `video` modes (`Args.mode`) and BDDL-file-based task
  selection instead of just benchmark suite/index.
- **`run_experiments.py`** — batch-runs a yaml's `experiments:` list against
  `scripts/serve_policy.py` (starting/stopping the server automatically whenever the
  checkpoint changes), scores each rollout using the yaml's `scoring:` block via
  `score_rollouts.py`, and writes `<stem>_results.csv` + `<stem>_episodes.txt` next to
  the input yaml.
- **`experiments/sim1.yaml` .. `sim6.yaml`** (+ `*_video.yaml` variants) — the actual
  batch configs used for the six memory tasks' ablations in the paper (text / keyframe /
  no-memory arms, plus a separate video-encoder variant per task). Run with:

  ```bash
  uv run examples/libero/run_experiments.py examples/libero/experiments/sim1.yaml
  uv run examples/libero/run_experiments.py examples/libero/experiments/sim1.yaml --dry-run
  ```

  Each yaml's `server.dir` points at a checkpoint path from our training runs — point it
  at your own checkpoint (see [training your own model](../../README.md#fine-tuning-base-models-on-your-own-data))
  before running.
- **`label_dataset_libero.py`** — generates the per-timestep event labels + event-history
  text that the memory-conditioned training pipeline supervises on, from recorded LIBERO
  episodes.
- **`convert_libero_data_to_lerobot.py`** / **`convert_libero_hdf5_to_lerobot.py`** —
  convert raw LIBERO(-Mem) HDF5 demonstrations to a LeRobot dataset (see each script's
  module docstring for which source format it expects).
