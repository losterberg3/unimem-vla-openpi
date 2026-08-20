# Latency benchmarks

Measures the latency impact of giving π0.5 access to visual history, across
three modes:

- **Base Single-Frame (π0.5)** — no video encoder, no history at all.
- **Naive Video (π0.5 + VE)** — a video encoder that re-encodes the full stack
  of context frames through spatial + temporal SigLIP attention on every call.
- **Ours (Keyframe Caching)** — the same video encoder architecture, but with a
  server-side SigLIP hidden-state cache: only the current frame is encoded each
  step, with each temporal attention layer attending against a cached raw
  hidden state (pre-temporal-PE, pre-projection) from prior frames instead of
  re-running the vision encoder on them. This is not a standard transformer
  KV cache — the cached tensor is the pre-projection residual-stream
  activation, not the projected key/value tensors themselves; each temporal
  layer still computes its own Q/K/V fresh from the cache every call, and the
  positional embedding is re-applied each time based on the frame's current
  relative age.

The x-axis is context length in images (keyframes).

## Scripts

| Script | Cameras |
|---|---|
| `benchmark_two_cameras.py` | 2 (real xarm rig: `base_0_rgb`, `left_wrist_0_rgb`) |
| `benchmark_four_cameras.py` | 4 (synthetic — no real rig has this many) |

Both scripts print median/std latency (ms) per (mode, K) point to stdout; there
is no plotting code in this release. Model hyperparameters are hardcoded to
match the `xarm_mem8_infer` TrainConfig (`src/openpi/training/config.py`) —
`pi05=True`, `action_dim=32`, `action_horizon=50`, `gemma_2b_lora` +
`gemma_300m_lora`, `event_tracking=True` — so the numbers are directly
comparable to what that config produces live. Both scripts use random-init
weights (no checkpoint needed): latency depends on tensor shapes and the
compute graph, not on weight values.

`benchmark_four_cameras.py` exists because camera count and keyframe count are
orthogonal bottlenecks: caching only avoids *re-encoding stale frames*, it
does nothing for the cost of encoding more *current* cameras every step — both
naive and cached pay that in full. No real xarm rig actually has 4 cameras;
this uses two synthetic extra camera keys in the observation dict (nothing
else needs to change — `embed_prefix` in `pi0.py` loops over cameras
generically, and Gemma uses RoPE, not a fixed position-embedding table, so a
longer image-token prefix just costs more compute, not a hard sequence-length
ceiling).

## What's actually measured

Both scripts spin up a real `Policy` wrapped in a `WebsocketPolicyServer`, and
talk to it with a real `WebsocketClientPolicy` over localhost — the actual
serving stack used in production, not a bare model function call. The
reported number is the `server_timing.infer_ms` field the server returns with
every response: a purely server-side measurement (no client-side scheduling
jitter) that covers the whole `Policy.infer()` call — input transform, the
SigLIP cache-encode step where applicable, `sample_actions_event`, and output
packaging.

## Requirements

A machine with a working GPU. Both scripts build and run a ~2.3B-parameter
model through JAX/XLA; on a CPU-only machine `jax` will raise
`FAILED_PRECONDITION: No visible GPU devices`.

## Running

Each point runs in its own subprocess (clean GPU memory + a fresh websocket
server per point), so the two scripts are independent and can be run in
either order:

```bash
uv run benchmarks/benchmark_two_cameras.py
uv run benchmarks/benchmark_four_cameras.py
```
