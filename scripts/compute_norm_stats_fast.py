"""Compute norm stats directly from parquet, skipping image decoding.

compute_norm_stats.py only uses `state` and `actions`, but it pulls every sample through
the full dataset pipeline — which decodes the current frame AND the event keyframes for
every image key. 

--verify builds the real pipeline and checks agreement on a sample before writing, so a
mismatch in the chunking convention is caught rather than silently poisoning training.
Run it the first time on a new dataset shape; skip it afterwards.

    uv run scripts/compute_norm_stats_fast.py xarm_mem10_coruscant --verify
"""

import argparse
import pathlib

import numpy as np
import pandas as pd

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def episode_files(repo_id: str) -> list[pathlib.Path]:
    root = pathlib.Path.home() / ".cache/huggingface/lerobot" / repo_id
    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {root}")
    return files


def collect(repo_ids: list[str], horizon: int, transform) -> tuple[np.ndarray, np.ndarray]:
    """Read the numeric columns, build the action chunk, then run the REAL transform stack.

    Reimplementing the transforms would silently drift when they change — this config
    already applies DeltaActions(mask=(True,)*6+(False,)), which converts the first six
    action dims to deltas from state. So instead we feed parquet rows through the actual
    transform objects with a 1x1 dummy image: the image path is exercised and discarded,
    and state/actions come out exactly as training sees them.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    def image_columns(path) -> list[str]:
        """Encoded images are struct<bytes, path>. Detect them from the schema rather than
        hardcoding names, so this works for any dataset, not just xarm."""
        s = pq.ParquetFile(path).schema_arrow
        out = []
        for n in s.names:
            t = s.field(n).type
            if pa.types.is_struct(t) and any(f.name == "bytes" for f in t):
                out.append(n)
        return out

    dummy = np.zeros((1, 1, 3), dtype=np.uint8)
    states, actions = [], []
    for rid in repo_ids:
        for f in episode_files(rid):
            IMG = image_columns(f)
            cols = [c for c in pq.ParquetFile(f).schema_arrow.names if c not in IMG]
            df = pd.read_parquet(f, columns=cols)
            a = np.stack(df["actions"].values).astype(np.float32)
            n = len(a)
            # delta_timestamps yields frames t..t+horizon-1, clamped at the episode end.
            # --verify confirms this against the real loader.
            idx = np.minimum(np.arange(n)[:, None] + np.arange(horizon)[None, :], n - 1)
            chunks = a[idx]
            recs = df.to_dict("records")
            for i in range(n):
                row = dict(recs[i])
                row["actions"] = chunks[i]
                for k in IMG:
                    row[k] = dummy
                # PromptFromLeRobotTask adds this downstream of the parquet; it does not
                # affect state/actions, so any string will do.
                row.setdefault("prompt", "")
                out = transform(row)
                states.append(np.asarray(out["state"], dtype=np.float32))
                actions.append(np.asarray(out["actions"], dtype=np.float32))
    return np.stack(states), np.stack(actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    # Accept both forms so this is a drop-in for compute_norm_stats.py, which uses
    # tyro and is invoked as `--config-name <name>` throughout cluster/.
    ap.add_argument("config_name", nargs="?", default=None)
    ap.add_argument("--config-name", dest="config_name_flag", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check against the real pipeline on --verify-n samples first; "
                         "exits non-zero on mismatch rather than writing bad stats")
    ap.add_argument("--verify-n", type=int, default=64)
    args = ap.parse_args()
    config_name = args.config_name_flag or args.config_name
    if not config_name:
        raise SystemExit("provide a config name, positionally or via --config-name")

    cfg = _config.get_config(config_name)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    repo_ids = list(dc.repo_ids) if dc.repo_ids else [dc.repo_id]
    horizon = cfg.model.action_horizon
    print(f"repos: {repo_ids}   action_horizon: {horizon}")

    import openpi.transforms as transforms
    chain = transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])
    print("transforms: " + " -> ".join(
        type(t).__name__ for t in (*dc.repack_transforms.inputs, *dc.data_transforms.inputs)))
    S, A = collect(repo_ids, horizon, chain)
    print(f"frames: {len(S)}   state {S.shape[1:]}   actions {A.shape[1:]}")

    if args.verify:
        import openpi.training.data_loader as dl
        import openpi.transforms as transforms

        class RemoveStrings(transforms.DataTransformFn):
            def __call__(self, x):
                return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

        print("building the real pipeline to cross-check ...")
        ds = dl.create_torch_dataset(dc, horizon, cfg.model)
        tds = dl.TransformedDataset(
            ds, [*dc.repack_transforms.inputs, *dc.data_transforms.inputs, RemoveStrings()]
        )
        if len(tds) != len(S):
            raise SystemExit(f"length mismatch: pipeline {len(tds)} vs fast path {len(S)}")
        rng = np.random.RandomState(0)
        idx = rng.choice(len(tds), size=min(args.verify_n, len(tds)), replace=False)
        ds_err = da_err = 0.0
        for i in idx:
            ref = tds[int(i)]
            ds_err = max(ds_err, float(np.abs(np.asarray(ref["state"]) - S[i]).max()))
            da_err = max(da_err, float(np.abs(np.asarray(ref["actions"]) - A[i]).max()))
        print(f"  max |state| diff over {len(idx)} samples: {ds_err:.3e}")
        print(f"  max |actions| diff:                       {da_err:.3e}")
        tol = 1e-4
        if ds_err > tol or da_err > tol:
            raise SystemExit("MISMATCH — refusing to write. The fast path does not reproduce "
                             "the pipeline; fix the chunking convention before trusting this.")
        print("  MATCH — fast path reproduces the pipeline exactly")

    stats = {}
    for key, arr in (("state", S), ("actions", A.reshape(-1, A.shape[-1]))):
        rs = normalize.RunningStats()
        for i in range(0, len(arr), 8192):
            rs.update(arr[i:i + 8192])
        stats[key] = rs.get_statistics()

    asset_id = dc.asset_id or dc.repo_id
    out = cfg.assets_dirs / asset_id
    print(f"writing stats to: {out}")
    normalize.save(out, stats)


if __name__ == "__main__":
    main()
