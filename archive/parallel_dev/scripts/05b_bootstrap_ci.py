import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from hmm_agents.config import load_config
from hmm_agents.evaluate import COMPARISONS

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--eval-split", default="val", choices=["val"])
ap.add_argument("--tag", default="early", help="'early' (script 05) or 'cc' (script 06)")
args = ap.parse_args()
cfg = load_config(args.config)
res = cfg["paths"]["results"]
n_boot = cfg["early"].get("n_boot", 1000)
rng = np.random.default_rng(cfg["seed"])


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    r = rankdata(p)
    return (r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


preds = pd.read_parquet(res / f"{args.tag}_predictions_{args.eval_split}.parquet")
ci_rows, diff_rows = [], []
for t, sub in preds.groupby("t"):
    wide = sub.pivot(index="run_id", columns="features", values="p_fail")
    meta = sub.drop_duplicates("run_id").set_index("run_id").loc[wide.index]
    sets = list(wide.columns)
    P, y = wide.to_numpy(), meta.failed.to_numpy().astype(bool)
    codes, issues = pd.factorize(meta.instance_id)
    groups = [np.flatnonzero(codes == i) for i in range(len(issues))]

    point = np.array([auroc(y, P[:, j]) for j in range(len(sets))])
    boot = np.full((n_boot, len(sets)), np.nan)
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        yb = y[idx]
        if 0 < yb.sum() < len(yb):
            boot[b] = [auroc(yb, P[idx, j]) for j in range(len(sets))]
    boot = boot[~np.isnan(boot).any(1)]

    for j, s in enumerate(sets):
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        ci_rows.append({"t": t, "features": s, "auroc": point[j], "ci_lo": lo, "ci_hi": hi,
                        "n_runs": len(y), "n_issues": len(groups), "n_solved": int((~y).sum())})
    for a, b_ in COMPARISONS:
        if a not in sets or b_ not in sets:
            continue
        ja, jb = sets.index(a), sets.index(b_)
        d = boot[:, ja] - boot[:, jb]
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff_rows.append({"t": t, "comparison": f"{a} - {b_}", "diff": point[ja] - point[jb],
                          "ci_lo": lo, "ci_hi": hi, "share_boot_le_0": float((d <= 0).mean()),
                          "significant": bool(lo > 0 or hi < 0)})

ci, diff = pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)
ci.to_csv(res / f"{args.tag}_prediction_ci_{args.eval_split}.csv", index=False)
diff.to_csv(res / f"{args.tag}_prediction_diff_{args.eval_split}.csv", index=False)

pd.set_option("display.width", 200)
print(f"AUROC with 95% CI ({n_boot} issue-level bootstrap resamples, {args.eval_split}):")
ci["cell"] = ci.apply(lambda r: f"{r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
print(ci.pivot(index="t", columns="features", values="cell").to_string())
print("\nPaired differences (A - B), 95% CI; * = CI excludes 0:")
diff["cell"] = diff.apply(lambda r: f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                          + (" *" if r.significant else ""), axis=1)
print(diff.pivot(index="t", columns="comparison", values="cell").to_string())
