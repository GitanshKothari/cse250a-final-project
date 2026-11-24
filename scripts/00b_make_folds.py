import argparse

import pandas as pd

from hmm_agents.config import ROOT, load_config
from hmm_agents.splits import build_folds

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)

out = ROOT / cfg["split"]["folds_file"]
if out.exists() and not args.force:
    raise SystemExit(f"{out} already exists (folds are frozen).")
tab = pd.read_csv(ROOT / cfg["split"]["file"])
folds = build_folds(tab, cfg["cv"]["n_folds"], cfg["seed"])
folds.to_csv(out, index=False)

m = folds.merge(tab, on="instance_id")
print(f"Saved {len(folds):,} train issues in {cfg['cv']['n_folds']} folds -> {out}")
print(m.groupby("fold").agg(issues=("instance_id", "size"), runs=("n_runs", "sum"),
                            solve_rate=("solve_rate", "mean")).round(3))
