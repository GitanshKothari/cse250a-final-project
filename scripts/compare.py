import argparse
from pathlib import Path

import pandas as pd

from hmm_agents.config import ROOT, load_config
from hmm_agents.evaluate import paired_diff, training_repos
from hmm_agents.splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--config", default=None)
ap.add_argument("--cohort", default=None, help="only print this cohort (default: all)")
args = ap.parse_args()
cfg = load_config(args.config)


def load(spec: str) -> tuple[Path, str, pd.DataFrame]:
    d, method = spec.rsplit(":", 1)
    d = Path(d)
    pred = pd.read_parquet(d / "predictions.parquet")
    if method not in set(pred.method):
        raise SystemExit(f"{method!r} not in {d}; available: {sorted(set(pred.method))}")
    return d, method, pred[pred.method == method]


(da, ma, pa), (db, mb, pb) = load(args.a), load(args.b)
if da.name != db.name:
    raise SystemExit(f"cannot pair a {da.name} result with a {db.name} result")
mode = da.name
la = f"A:{da.parent.name}/{ma}" if da != db else ma
lb = f"B:{db.parent.name}/{mb}" if da != db else mb
if la == lb:
    raise SystemExit("A and B are the same predictions")
pred = pd.concat([pa.assign(method=la), pb.assign(method=lb)], ignore_index=True)
seen = training_repos(ROOT / cfg["split"]["file"])
if mode == "cv":
    pred["fold"] = pred["instance_id"].map(load_folds(ROOT / cfg["split"]["folds_file"]))
    seen = None
ec = cfg["eval"]
out = paired_diff(pred, la, lb, ec["n_boot"], cfg["seed"], ec["fixed_min_steps"], seen)
if args.cohort:
    out = out[out.cohort == args.cohort]
cols = ["cohort", "t", "n_runs", "d_auroc", "d_auroc_lo", "d_auroc_hi", "p_b_better_auroc",
        "d_auprc_success", "d_auprc_success_lo", "d_auprc_success_hi", "d_brier"]
print(f"B - A   (A = {la}, B = {lb})")
print(out[cols].round(3).to_string(index=False))
safe = lambda s: s.replace("/", "_").replace(":", "_")
out.to_csv(db / f"compare_{safe(la)}_vs_{safe(lb)}.csv", index=False)
