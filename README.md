Forked from https://github.com/Physical-Intelligence/openpi

## UniMem

This repository accompanies our paper **UniMem: Unifying Multimodal Memory and Control
for VLAs** (Osterberg, Wang, Schwager — Stanford University, 2026, preprint).

If you use this code, please cite:
```bibtex
@misc{osterberg2026unimem,
  title        = {UniMem: Unifying Multimodal Memory and Control for VLAs},
  author       = {Osterberg, Lars and Wang, Maggie and Schwager, Mac},
  year         = {2026},
  institution  = {Stanford University},
  note         = {Preprint}
}
```

## Fork Overview

This fork adds event-memory conditioning to π₀/π₀.₅, with model and training changes
under `src/openpi/` (see the pipeline overview below, and the annotated file tree in
["Repository Structure"](#repository-structure-only-files-weve-touched-or-added) for
exactly which files changed), plus two evaluation domains used in the paper:
- [`examples/libero`](examples/libero) — memory-conditioned LIBERO simulation experiments
- [`examples/xarm`](examples/xarm) — memory-conditioned real-robot xArm experiments

### Pipeline overview

This fork adds "event memory" to base π₀/π₀.₅: the policy can detect semantic events
(e.g. "grabbed box") and, optionally, attend back to frames from past events instead of
only the current one. Everything else — the flow-matching action head, LoRA/full
fine-tuning, LeRobot data loading, the websocket serving stack — is unmodified upstream
openpi (see "openpi" below for that base architecture). Two independent pieces on top:

1. **Event tracking** (`Pi0Config.event_tracking`) adds an MLP classification head
   (`Pi0.phase_head`, defined by `EventHead` in `models/event_head.py`) on the pooled
   prefix representation, trained with an auxiliary cross-entropy loss
   (`Pi0.compute_loss_event`) against a per-timestep `labels` column your dataset
   provides. At inference, `Policy.infer()` returns the predicted event id alongside
   `actions` (see `models/pi0.py`, `policies/policy.py`).
2. **Video encoder** (`Pi0Config.video_encoder`) makes SigLIP ingest `num_frames` per
   camera instead of 1, via temporal attention (`models/siglip.py`'s
   `TemporalStrideBlock`/`VideoEncoder`). This is purely a model-architecture flag —
   it says nothing about *which* frames get selected, which is a training-data-config
   choice with two very different flavors:
   - **Fixed-stride ("naive video")** — a constant time stride ending at the current
     frame. Never combines with event tracking. Served by having the *client* stack
     the whole frame history itself on every call.
   - **Keyframes** — the actual past *event*-transition frames, via
     `training/data_loader.py`'s `EventMemoryDataset`. Always paired with event
     tracking (frame selection depends on the event labels). Served incrementally: the
     client sends only the current frame, and `Policy` maintains a rolling
     hidden-state cache (`models/siglip_hidden_cache.py`) across calls, which must be
     reset once per rollout (`Policy.reset_cache()`).

`src/openpi/training/config.py` has one `DataConfigFactory` subclass per (robot,
frame-selection) combination — see "Training your own UniMem policy" below for
the full list and which ones are templates vs. this fork's actual experiment sweep.

### A note on terminology

The paper — and all user-facing code, comments, and configs in this fork — uses
**"event"** for a semantic transition point in a task (e.g. "grabbed box", "tapped left
basket"). Internally, exactly two things still say **"phase"**: the `phase_history`
observation key sent to the served model, and the `self.phase_head` attribute on `Pi0`
(defined by the `EventHead` class in `models/event_head.py` — the class, file, and
`compute_loss_event` method are all renamed; only the attribute assignment itself keeps
the old name, and `phase_head_lr_multiplier` is named after it). Both are baked into the
wire protocol, the training data schema, or the checkpoint's own parameter tree for every
already-trained checkpoint (LIBERO and xArm) — renaming either would silently break
loading those checkpoints, so "phase" is kept there deliberately as a legacy synonym for
"event." Everything else that used to say "phase" was plain metadata or a training-time
identifier, not baked into a parameter tree or wire format, so it's all been renamed:
`Pi0Config.phase_tracking` → `event_tracking` (with a one-line backward-compatibility
shim in `policy_config.py` for checkpoints whose `train_shape.json` predates the rename),
the `LeRobotLiberoPhaseDataConfig`/`LeRobotXarmPhaseDataConfig` family → `...Event...DataConfig`,
and the `phase_id`/`skip_first_phase_ids`/`upsample_after_phase_id` fields in
`training/config.py` and `training/data_loader.py` → `event_id`/`skip_first_event_ids`/
`upsample_after_event_id`.

## Repository Structure (Only Files We've Touched or Added)

```
openpi/
├── README.md                                 # this file — paper info, terminology note (event vs. phase), and this file tree
├── pyproject.toml                             # uv-managed Python environment (added hardware specific libraries, e.g. librealsense and xArmSDK)
├── uv.lock                                    # locked dependency versions
│
├── src/
│   └── openpi/                                # main Python package
│       ├── transforms.py                      # DataTransformFn pipeline pieces shared by training + inference, added event language to tokenizer and normalization divide-by-zero bug fix
│       │
│       ├── models/                            # model architectures (JAX / Flax NNX)
│       │   ├── model.py                       # necessary SigLIP conversion for VideoEncoder architecture and param check/alignment for the additional event classifier
│       │   ├── pi0.py                         # Added optional event-classification head (EventHead + compute_loss_event) and video encoder (temporal attention over num_frames)
│       │   ├── pi0_config.py                  # Pi0Config dataclass (action_dim, video_encoder, event_tracking, num_frames, ...)
│       │   ├── event_head.py                  # EventHead: MLP classifier on pooled prefix repr, for event labels (self.phase_head attribute name kept — see terminology note above)
│       │   ├── siglip.py                      # SigLIP vision encoder (spatial ViT); temporal attention (TemporalStrideBlock/VideoEncoder) added on top of upstream
│       │   ├── siglip_hidden_cache.py         # SigLIP + causal temporal hidden-state cache, for incremental video/keyframe processing at serve time
│       │   └── tokenizer.py                   # Event language added to tokenize function
│       │
│       ├── policies/                          # per-robot observation/action transforms + the served Policy wrapper
│       │   ├── policy.py                      # Policy: wraps a model + transforms, runs inference, packages the event-classification output and hidden-state-cache reset/slide signals
│       │   ├── policy_config.py               # create_trained_policy(): builds a Policy from a TrainConfig + checkpoint dir; validates config against how the checkpoint was trained
│       │   ├── libero_policy.py               # LiberoInputs/LiberoOutputs — this fork's LIBERO-memory observation/action mapping
│       │   └── xarm_policy.py                 # XarmInputs/XarmOutputs — this fork's xArm-memory observation/action mapping
│       │
│       └── training/
│           ├── config.py                      # TrainConfig + DataConfigFactory registry — unimem_example_* templates plus every real libero_mem*/xarm_mem* config, all live here
│           ├── data_loader.py                 # LeRobot dataset loading/batching, event-history + keyframe sampling, text/event dropout
│           ├── weight_loaders.py              # base-checkpoint weight loading for fine-tuning (incl. missing-param regex for lora/phase_head/temporal)
│           └── lerobot_hf_patch.py            # patches a LeRobot HF-transform quirk before LeRobotDataset import
│
├── scripts/                                   # CLI entry points
│   ├── train.py                               # JAX fine-tuning entry point
│   ├── train_accum_steps.py                   # train.py variant with gradient accumulation
│   ├── train_pytorch.py                       # PyTorch fine-tuning entry point (mirrors train.py)
│   ├── serve_policy.py                        # loads a checkpoint and starts the WebsocketPolicyServer
│   ├── compute_norm_stats.py                  # computes state/action normalization stats for a TrainConfig
│   ├── compute_norm_stats_fast.py             # faster norm-stats variant that skips image decoding (state/actions only)
│   └── docker/                                # Dockerfile + install scripts for serve_policy
│
├── examples/
│   ├── libero/                                # ★ this fork's LIBERO-memory simulation experiments
│   │   ├── README.md                          # setup/usage (Docker + local), results table, memory-experiment workflow (run_experiments.py + experiments/ yamls)
│   │   ├── libero_inference.py                # event-memory-conditioned rollout client (Args, run_event_inference) — this fork's main inference script
│   │   ├── run_experiments.py                 # batch-runs a YAML's experiments against serve_policy, scores them, writes <stem>_results.csv + <stem>_episodes.txt
│   │   ├── score_rollouts.py                  # scoring functions: grab_place_xy, drop_tap_match, object_return_xy, initial_obj_to_event_xy, event_completion
│   │   ├── label_dataset_libero.py            # generates per-timestep event labels + event-history text for recorded LIBERO episodes (dataset prep; Claude-generated)
│   │   ├── video_annotation.py                # shared rollout-frame text-overlay renderer
│   │   └── experiments/                       # sim1.yaml..sim6.yaml + *_video.yaml — one batch config per LIBERO memory task (text/keyframe/no_memory ablations + video-encoder variant)
│   │
│   └── xarm/                                  # ★ this fork's real-robot xArm memory experiments
│       ├── README.md                          # hardware assumptions (one xArm + 2 RealSense cams) and how to adapt to your own rig
│       ├── xarm_inference.py                  # event-memory-conditioned rollout client for the physical xArm using real time chunking; arm IP/camera serials/home pose are Args fields, not hardcoded
│       ├── run_experiments.py                 # batch-runs a YAML's experiments against serve_policy; console latency summary only — no scoring/CSV, hardware rollouts aren't auto-scorable
│       ├── label_dataset_xarm.py              # generates per-timestep event labels + event-history text for recorded xArm hardware episodes (Claude-generated)
│       ├── demo_collection.py                 # teleoperated demo recording → LeRobot dataset (arm IP/camera serials are CLI flags; repo name/task description are per-session constants in the file)
│       └── experiments/                       # mem7.yaml..mem10.yaml — one batch config per xArm memory task
│
└── benchmarks/                                # latency benchmarking: cost of visual history (base / naive video / keyframe-caching)
    ├── README.md                              # methodology and the three comparison modes
    ├── benchmark_two_cameras.py               # latency vs. context length, 2-camera xArm rig config
    └── benchmark_four_cameras.py              # same, replicated for a 4-camera rig
```

## Getting Started with UniMem

### 1. Curating your dataset

Clone and sync this repo following the usual steps below (see "Installation" further down for the full version). Make any changes to `pyproject.toml` with packages you need in your hardware setup (RealSense and xArm packages are currently included).

```bash
git clone --recurse-submodules <your-fork-url>
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Decide on your robot memory task. Collect demonstrations in LeRobot v2.1 format with the same observation structure as examples/xarm/demo_collection.py (an optional human event marker is included in case you want to label during demonstrations). Adapt this demonstration collection script to your own hardware — arm IP and camera serials are CLI flags there; the dataset repo name and task description are per-session constants at the top of the file.

```bash
uv run examples/xarm/demo_collection.py \
    --arm-ip <your-arm-ip> --camera-serial-external <serial> --camera-serial-wrist <serial>
```

Using an agentic framework, add a new labeling function to examples/xarm/label_dataset_xarm.py and make any other adjustments for your setup. An example prompt to an agent for our TapScoopPour task is shown below.

> ### Example Claude Sonnet 5.0 Prompt
>
> Write a labeling function for our xArm demonstration dataset. The task being demonstrated is: **a human taps one cup, and the robot then puts a single scoop of beans into that cup.**
>
> **Input.** One LeRobot parquet per episode, recorded at 20 fps. Per frame you have the end-effector pose (`xyz` in mm, roll/pitch/yaw), a gripper channel, and a `human_event` column that is `1.0` on the single frame where the operator pressed the tap key during teleoperation.
>
> **Output.** The script should label each frame in the dataset with both a discrete event id from a vocabulary set and a textual memory string.
>
> **Vocabulary.** Five events, each occurring exactly once per episode:
> `{0: "human tap", 1: "grabbed spoon", 2: "scooped beans", 3: "poured beans", 4: "placed spoon"}`. Frames belonging to no event get the null target `-1`.
>
> **Detection and labeling.**
> - **0 — human tap:** first frame where `human_event == 1.0`. Warn if there is more than one marker and use the first.
> - **1 — grabbed spoon:** the fully-closed gripper plateau of the first gripper close occurring after the tap.
> - **2 — scooped beans:** scan for the first 40-frame window satisfying all of: roll std < 5°, mean roll within ±20° of 180° (either sign), yaw std < 5°, mean `z` < 255 mm, and pitch increasing on at least 60% of frames (sustained straightening while low in the bowl). Label the end of the 40-frame window with this event.
> - **3 — poured beans:** the frame of minimum roll.
> - **4 — placed spoon:** first gripper open after the scoop.
>
> Label the window of frames surrounding the event, starting 5 frames before and ending 20 frames after the detection.
>
> **Textual memory.** An event's phrase becomes visible in the memory string only once the frame is no longer labeled with that event. Render as `"History: human tap, grabbed spoon, ..."` and `"History: none"` before the first event is visible.

Run the labeling script to get new event and textual memory columns in your dataset — `--write-parquet` is what actually persists them (without it, it's a dry run that just prints/renders debug videos):

```bash
uv run examples/xarm/label_dataset_xarm.py \
    --dataset-root ~/.cache/huggingface/lerobot/<your-hf-username>/<dataset_name>/data/chunk-000 \
    --task-name <your_task_name> \
    --write-parquet
```

These are the keys "labels" and "phase_history", respectively, in the LeRobot dataset. You now have your memory dataset curated and are ready to start training!

### 2. Training your own UniMem policy

This fork adds an optional **event-tracking head** (detects semantic events like "grabbed
box" or "tapped left basket" — see the [terminology note](#a-note-on-terminology) above)
and an optional **event-conditioned video encoder** (lets the policy attend back to past
event frames instead of only the current one) on top of the base π₀.₅ fine-tuning flow
below. `src/openpi/training/config.py` has two kinds of configs for this:

- **`unimem_example_libero`, `unimem_example_xarm`, `unimem_example_libero_keyframe`,
  `unimem_example_libero_video`** — heavily-commented templates. Copy whichever is
  closest to your setup, swap in your dataset's `repo_id`, and tune the knobs from
  there — every field is explained inline (LoRA vs. full fine-tune,
  `phase_head_lr_multiplier`, `event_dropout_prob`/`text_dropout_prob`,
  `event_frame_window`, `upsample_after_event_id`, ...).

  A common point of confusion: `video_encoder=True` is a **model** flag (turns on
  SigLIP temporal attention over T frames — see `Pi0Config.video_encoder`'s docstring)
  and is independent of event tracking. It does NOT mean "video always implies
  events," and the two data pipelines that use it are not symmetric:
  - `unimem_example_libero_video` (`LeRobotLiberoDataConfig`, fixed
    `frame_stride_sec`) — video with **no event tracking at all**. A plain "does
    temporal context help" baseline; never loads `labels`/`phase_history`.
  - `unimem_example_libero_keyframe` (`LeRobotLiberoEventKeyframeDataConfig`, via
    `EventMemoryDataset`) — video **conditioned on actual past event frames**,
    trained together with the event-classification head. This is the only
    `video_encoder=True` config that also does event tracking.

  Video and keyframes train the identical SigLIP architecture; only the
  frame-selection pipeline (and whether an event head is even present) differs. The
  single-frame `unimem_example_libero`/`unimem_example_xarm` configs (event tracking,
  no video encoder at all) are a third, separate case — see
  `LeRobotLiberoEventDataConfig`'s docstring in `config.py` for why none of these four
  are interchangeable.
- **`libero_mem1`..`libero_mem6` and `xarm_mem7`..`xarm_mem10`** (plus their
  `_video`/`_finetune`/`_coruscant`/`_no_memory`/`_text_only`/`_keyframe_only`/`_infer`
  variants) — the actual sweep used to produce this fork's checkpoints, kept as-is
  for reproducibility rather than as a template to copy.

To train your own event-memory policy:

1. Your LeRobot dataset needs two extra columns beyond the base fine-tuning flow (should already be done in the previous section): an
   integer `labels` column (-1 = unlabeled, 0..N-1 = your event class ids) and a
   `phase_history` text column (a running summary like `"History: grabbed box"`, built
   from completed events — see the "note on terminology" above for why the key keeps
   the name `phase_history`). [`examples/libero/label_dataset_libero.py`](examples/libero/label_dataset_libero.py)
   and [`examples/xarm/label_dataset_xarm.py`](examples/xarm/label_dataset_xarm.py) show
   how we generated both from recorded episodes for LIBERO and xArm respectively —
   copy and adapt whichever is closer to your setup.
2. Copy whichever `unimem_example_*` config above is closest to your setup into
   `training/config.py` and point it at your dataset (`repo_id`, `AssetsConfig.asset_id`).
3. Compute norm stats, then train — same commands as any other π₀.₅ checkpoint (see
   "Fine-Tuning Base Models on Your Own Data" below), just with your config's name:

   ```bash
   uv run scripts/compute_norm_stats.py --config-name <your_config_name>
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <your_config_name> --exp-name=my_experiment --overwrite
   ```

### 3. Running your UniMem policy

To run your policy, edit the hardware specific variables and task specific vocabulary inside examples/xarm/xarm_inference.py. Write your experiment yaml file under experiments/ — copy one of the existing ones (e.g. `mem7.yaml`) and point `server.config`/`server.dir` at your own checkpoint; you shouldn't have to change much else. Note that we use gripper closing threshold and snap logic; this is just so that the gripper doesn't twitch when grasping. `run_experiments.py` starts the policy server itself (from the yaml's `server` block), so this one command is all you need to start sending commands to your robot:

```bash
uv run examples/xarm/run_experiments.py examples/xarm/experiments/mem7.yaml
```

Inference Note: `Policy` auto-detects `model.event_tracking` and returns an
   `event_id` field (predicted event-class probabilities) alongside `actions` — see
   [`src/openpi/policies/policy.py`](src/openpi/policies/policy.py). How you serve the
   video encoder depends on which frame-selection pipeline you trained with (see
   "Pipeline overview" above) — the two are NOT interchangeable at serve time:
   - **Keyframe-trained**: send only the current frame each call, and call
     `Policy.reset_cache()` once per rollout. `Policy` maintains the rolling
     hidden-state cache (see
     [`src/openpi/models/siglip_hidden_cache.py`](src/openpi/models/siglip_hidden_cache.py))
     internally; skipping `reset_cache()` carries stale history across episodes, and
     feeding it single frames without ever sliding just repeats the current frame.
   - **Fixed-stride ("naive video")-trained**: `Policy`'s cache is not used at all —
     stack the last `num_frames` frames yourself (matching the `frame_stride_sec`
     spacing you trained with) and send the whole stack every call, same as a
     single-frame model just with a `(T, H, W, C)` image instead of `(H, W, C)`.

### 4. Helpful Tips

- Make sure the same event is never labeled twice in a row. This shouldn't happen
  naturally anyway: returning to a cyclical subtask's starting point requires reversing
  your forward action, which is itself a distinct event. Our inference scripts already
  enforce this — repeats of the same event are never appended consecutively to the
  textual memory.
- When training, consider upweighting decisive moments — the short windows where your
  policy's action depends on memory (e.g. which of two baskets to place into). These
  windows are often brief relative to the rest of a demonstration, but getting them
  right is what makes or breaks a memory policy.

The rest of openpi's original README.md is below.

# openpi

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, this repo contains three types of models:
- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based vision-language-action model (VLA).
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.
- the [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05), an upgraded version of π₀ with better open-world generalization trained with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation). Note that, in this repository, we currently only support the flow matching head for both $\pi_{0.5}$ training and inference.

For all models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

This is an experiment: $\pi_0$ was developed for our own robots, which differ from the widely used platforms such as [ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!

## Updates

- [Sept 2025] We released PyTorch support in openpi.
- [Sept 2025] We released pi05, an upgraded version of pi0 with better open-world generalization.
- [Sept 2025]: We have added an [improved idle filter](examples/droid/README_train.md#data-filtering) for DROID training.
- [Jun 2025]: We have added [instructions](examples/droid/README_train.md) for using `openpi` to train VLAs on the full [DROID dataset](https://droid-dataset.github.io/). This is an approximate open-source implementation of the training pipeline used to train pi0-FAST-DROID. 


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.

## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case    | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference   | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference   | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO      | Inference   | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID      | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`      |


By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.




## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the [LIBERO dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [π₀](src/openpi/training/config.py), [π₀-FAST](src/openpi/training/config.py), and [π₀.₅](src/openpi/training/config.py) on LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).

### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch Support

openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the LIBERO benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup
1. Make sure that you have the latest version of all dependencies installed: `uv sync`

2. Double check that you have transformers 4.53.2 installed: `uv pip show transformers`

3. Apply the transformers library patches:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

**WARNING**: With the default uv link mode (hardlink), this will permanently affect the transformers library in your uv cache, meaning the changes will survive reinstallations of transformers and could even propagate to other projects that use transformers. To fully undo this operation, you must run `uv cache clean transformers`.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**
1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**
1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If issues persist, check that you have the latest version of `uv` installed (`uv self update`). |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may way to consider disabling EMA.        |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                            |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                          |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                 |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try *uninstalling* system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts. |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                    |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                  |
| Diverging training loss                            | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround. |
