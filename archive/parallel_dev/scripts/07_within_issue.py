import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hmm_agents.config import load_config
from hmm_agents.evaluate import COMPARISONS, pair_counts

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--eval-split", default="val", choices=["val"])
ap.add_argument("--tag", default="early", help="'early' (script 05) or 'cc' (script 06)")
args = ap.parse_args()
cfg = load_config(args.config)
res = cfg["paths"]["results"]
n_boot = cfg["early"].get("n_boot", 1000)
rng = np.random.default_rng(cfg["seed"])


runs = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet",
                       columns=["run_id", "model_name"]).set_index("run_id")
preds = pd.read_parquet(res / f"{args.tag}_predictions_{args.eval_split}.parquet")

rows, diff_rows = [], []
for t, sub in preds.groupby("t"):
    wide = sub.pivot(index="run_id", columns="features", values="p_fail")
    meta = sub.drop_duplicates("run_id").set_index("run_id").loc[wide.index]
    sets = list(wide.columns)
    P, y = wide.to_numpy(), meta.failed.to_numpy().astype(bool)
    issue, _ = pd.factorize(meta.instance_id)
    model = runs.loc[wide.index, "model_name"].to_numpy()
    mixed = pd.Series(y).groupby(issue).transform(lambda v: 0 < v.mean() < 1).to_numpy()

    for j, s in enumerate(sets):
        rows.append({"t": t, "features": s, "variant": "all", "auroc": roc_auc_score(y, P[:, j]),
                     "n_issues": len(np.unique(issue)), "n_runs": len(y), "n_pairs": y.sum() * (~y).sum()})
        rows.append({"t": t, "features": s, "variant": "mixed_issues",
                     "auroc": roc_auc_score(y[mixed], P[mixed, j]),
                     "n_issues": len(np.unique(issue[mixed])), "n_runs": int(mixed.sum()),
                     "n_pairs": y[mixed].sum() * (~y[mixed]).sum()})

    groups = {"within_issue": issue,
              "within_issue_model": pd.factorize(pd.Series(issue).astype(str) + "|" + model)[0]}
    for variant, group in groups.items():
        C, N = pair_counts(P, y, group, issue)
        keep = np.flatnonzero(N > 0)
        C, N = C[keep], N[keep]
        point = C.sum(0) / N.sum()
        boot = np.empty((n_boot, len(sets)))
        for b in range(n_boot):
            idx = rng.integers(0, len(keep), len(keep))
            boot[b] = C[idx].sum(0) / N[idx].sum()
        n_runs = int(np.isin(issue, keep).sum())
        for j, s in enumerate(sets):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            rows.append({"t": t, "features": s, "variant": variant, "auroc": point[j],
                         "ci_lo": lo, "ci_hi": hi, "n_issues": len(keep), "n_runs": n_runs,
                         "n_pairs": int(N.sum())})
        for a, b_ in COMPARISONS:
            if a not in sets or b_ not in sets:
                continue
            ja, jb = sets.index(a), sets.index(b_)
            d = boot[:, ja] - boot[:, jb]
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff_rows.append({"t": t, "variant": variant, "comparison": f"{a} - {b_}",
                              "diff": point[ja] - point[jb], "ci_lo": lo, "ci_hi": hi,
                              "significant": bool(lo > 0 or hi < 0)})

out, diff = pd.DataFrame(rows), pd.DataFrame(diff_rows)
out.to_csv(res / f"{args.tag}_within_issue_{args.eval_split}.csv", index=False)
diff.to_csv(res / f"{args.tag}_within_issue_diff_{args.eval_split}.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
sizes = out[out.features == out.features.iloc[0]].pivot(index="t", columns="variant",
                                                         values=["n_issues", "n_runs", "n_pairs"])
print(f"Sample sizes ({args.eval_split}):\n{sizes.to_string()}\n")
for variant in ["all", "mixed_issues", "within_issue", "within_issue_model"]:
    v = out[out.variant == variant].copy()
    if "ci_lo" in v and v.ci_lo.notna().all():
        v["cell"] = v.apply(lambda r: f"{r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
    else:
        v["cell"] = v.auroc.map("{:.3f}".format)
    print(f"AUROC, {variant}:\n{v.pivot(index='t', columns='features', values='cell').to_string()}\n")
for variant in groups:
    d = diff[diff.variant == variant].copy()
    d["cell"] = d.apply(lambda r: f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                        + (" *" if r.significant else ""), axis=1)
    print(f"Paired differences, {variant} (A - B), 95% CI; * = CI excludes 0:")
    print(d.pivot(index="t", columns="comparison", values="cell").to_string() + "\n")
