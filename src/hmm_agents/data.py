from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .parsing import parse_trajectory

N_SHARDS = 12
RAW_COLUMNS = ["instance_id", "model_name", "target", "exit_status", "trajectory"]


def download_shards(repo_id: str, out_dir: Path, shards: list[int] | None) -> list[Path]:
    from huggingface_hub import hf_hub_download

    shards = list(range(N_SHARDS)) if shards is None else shards
    paths = []
    for s in shards:
        fname = f"data/train-{s:05d}-of-{N_SHARDS:05d}.parquet"
        paths.append(Path(hf_hub_download(repo_id, fname, repo_type="dataset", local_dir=out_dir)))
    return paths


def shard_index(path: Path) -> int:
    return int(Path(path).name.split("-")[1])


def build_runs_table(raw_files: list[Path], min_steps: int = 2) -> pd.DataFrame:
    rows = []
    for f in raw_files:
        pf = pq.ParquetFile(f)
        row_idx = shard_index(f) * 100_000
        for batch in tqdm(pf.iter_batches(batch_size=500, columns=RAW_COLUMNS),
                          total=int(np.ceil(pf.metadata.num_rows / 500)), desc=f.name):
            for r in batch.to_pylist():
                run_id, row_idx = row_idx, row_idx + 1
                steps, issue = parse_trajectory(r["trajectory"] or [], r["instance_id"])
                if len(steps) < min_steps:
                    continue
                rows.append({
                    "run_id": run_id,
                    "instance_id": r["instance_id"],
                    "model_name": r["model_name"],
                    "target": bool(r["target"]),
                    "exit_status": r["exit_status"],
                    "issue_len": len(issue),
                    "n_steps": len(steps),
                    "symbols": [s.symbol for s in steps],
                    "actions": [s.action for s in steps],
                    "errors": [s.error for s in steps],
                    "commands": [s.command for s in steps],
                    "viewed_files": [s.viewed_file or "" for s in steps],
                    "edited_files": [s.edited_file or "" for s in steps],
                    "test_passed": [s.test_passed for s in steps],
                    "no_match": [s.no_match for s in steps],
                    "success_msg": [s.success_msg for s in steps],
                })
    return pd.DataFrame(rows)


def stratified_subsample(df: pd.DataFrame, max_runs: int | None, seed: int) -> pd.DataFrame:
    if not max_runs or len(df) <= max_runs:
        return df
    rng = np.random.default_rng(seed)
    inst = df.groupby("instance_id")["target"].max()
    keep = []
    for label in (True, False):
        ids = inst[inst == label].index.to_numpy()
        rng.shuffle(ids)
        keep.append(ids[: int(len(ids) * max_runs / len(df))])
    return df[df["instance_id"].isin(np.concatenate(keep))].reset_index(drop=True)
