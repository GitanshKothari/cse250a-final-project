import argparse

import pandas as pd
import pyarrow.parquet as pq

from hmm_agents.config import ROOT, load_config
from hmm_agents.data import N_SHARDS
from hmm_agents.splits import build_split_table

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--force", action="store_true", help="overwrite an existing frozen table")
args = ap.parse_args()
cfg = load_config(args.config)
sc = cfg["split"]

out = ROOT / sc["file"]
if out.exists() and not args.force:
    raise SystemExit(f"{out} already exists. Splits are frozen; pass --force only if you "
                     "really mean to reshuffle (see dev_docs/04_frozen_splits.md).")

raw = sorted((cfg["paths"]["raw"] / "data").glob("train-*.parquet"))
if len(raw) != N_SHARDS:
    raise SystemExit(f"Need all {N_SHARDS} raw shards, found {len(raw)}. "
                     "Set data.shards: null and run scripts/01_download.py.")
meta = pd.concat([pq.read_table(f, columns=["instance_id", "target"]).to_pandas()
                  for f in raw], ignore_index=True)

tab = build_split_table(meta, sc["test"], sc["val"], sc["hmm_frac_of_train"], cfg["seed"])
tab.sort_values("instance_id").to_csv(out, index=False)

runs = meta.merge(tab[["instance_id", "split"]], on="instance_id")
print(f"Saved {len(tab):,} issues -> {out}\n")
summary = pd.DataFrame({
    "issues": tab.groupby("split").size(),
    "repos": tab.groupby("split")["repo"].nunique(),
    "runs": runs.groupby("split").size(),
    "run_success": runs.groupby("split")["target"].mean().round(3),
    "issue_solve_rate": tab.groupby("split")["solve_rate"].mean().round(3),
}).loc[["hmm_train", "clf_train", "val", "test"]]
print(summary)
train_repos = set(tab.loc[tab.split.isin(["hmm_train", "clf_train"]), "repo"])
for s in ["val", "test"]:
    part = tab[tab.split == s]
    print(f"{s}: {part.repo.isin(train_repos).mean():.1%} of issues come from a repo "
          "that also appears in the training splits")
