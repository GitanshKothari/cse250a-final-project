from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PRED_COLUMNS = ["method", "run_id", "instance_id", "t", "p_fail", "fail", "n_steps"]



def _group_sum(M: np.ndarray, G: sparse.csr_matrix) -> np.ndarray:
    return np.asarray((G.T @ M.T).T)


def _score_groups(score: np.ndarray) -> tuple[sparse.csr_matrix, int]:
    _, inv = np.unique(score, return_inverse=True)
    U = int(inv.max()) + 1
    G = sparse.csr_matrix((np.ones(len(score)), (np.arange(len(score)), inv)),
                          shape=(len(score), U))
    return G, U


def _auroc_num_den(score, y, W):
    G, _ = _score_groups(score)
    pos = _group_sum(W * y, G)
    neg = _group_sum(W * (1 - y), G)
    neg_below = np.cumsum(neg, axis=1) - neg
    return (pos * (neg_below + 0.5 * neg)).sum(axis=1), pos.sum(axis=1) * neg.sum(axis=1)


def weighted_auroc(score: np.ndarray, y: np.ndarray, W: np.ndarray,
                   strata: np.ndarray | None = None) -> np.ndarray:
    W = np.atleast_2d(W)
    if strata is None:
        num, den = _auroc_num_den(score, y, W)
    else:
        num = den = 0.0
        for s in np.unique(strata):
            m = strata == s
            n_, d_ = _auroc_num_den(score[m], y[m], W[:, m])
            num, den = num + n_, den + d_
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def weighted_auroc_within(score: np.ndarray, y: np.ndarray, W: np.ndarray,
                           groups: np.ndarray) -> np.ndarray:
    W = np.atleast_2d(W)
    _, gi = np.unique(groups, return_inverse=True)
    _, rk = np.unique(score, return_inverse=True)
    R = int(rk.max()) + 1 if len(rk) else 1
    ukey, inv = np.unique(gi.astype(np.int64) * R + rk, return_inverse=True)
    U, n = len(ukey), len(score)
    G = sparse.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, U))
    pos, neg = _group_sum(W * y, G), _group_sum(W * (1 - y), G)
    cneg = np.cumsum(neg, axis=1)
    ug = ukey // R
    first = np.r_[0, np.flatnonzero(np.diff(ug)) + 1]
    start = first[np.searchsorted(first, np.arange(U), side="right") - 1]
    neg_below = cneg - neg - (cneg[:, start] - neg[:, start])
    num = (pos * (neg_below + 0.5 * neg)).sum(axis=1)
    S = sparse.csr_matrix((np.ones(U), (np.arange(U), np.searchsorted(first, np.arange(U),
                          side="right") - 1)), shape=(U, len(first)))
    den = (_group_sum(pos, S) * _group_sum(neg, S)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def weighted_average_precision(score: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    W = np.atleast_2d(W)
    G, _ = _score_groups(-score)
    tp = _group_sum(W * y, G)
    fp = _group_sum(W * (1 - y), G)
    ctp, cfp = np.cumsum(tp, axis=1), np.cumsum(fp, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(ctp + cfp > 0, ctp / (ctp + cfp), 0.0)
        P = tp.sum(axis=1)
        return np.where(P > 0, (tp * prec).sum(axis=1) / P, np.nan)


def weighted_brier(p: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    W = np.atleast_2d(W)
    return (W * (p - y) ** 2).sum(axis=1) / W.sum(axis=1)


def _ci(b: np.ndarray) -> tuple[float, float]:
    b = b[~np.isnan(b)]
    return tuple(np.percentile(b, [2.5, 97.5])) if len(b) else (np.nan, np.nan)


def issue_bootstrap_weights(instance_ids: np.ndarray, n_boot: int, seed: int,
                            chunk: int = 250):
    issues, idx = np.unique(instance_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    n_iss = len(issues)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        counts = rng.multinomial(n_iss, np.full(n_iss, 1.0 / n_iss), size=b)
        yield counts[:, idx].astype(float)



METRICS = {
    "auroc": lambda p, y, W, s=None, i=None: weighted_auroc(p, y, W, s),
    "auroc_wi": lambda p, y, W, s=None, i=None: weighted_auroc_within(p, y, W, i),
    "auprc_success": lambda p, y, W, s=None, i=None: weighted_average_precision(1 - p, 1 - y, W),
    "brier": lambda p, y, W, s=None, i=None: weighted_brier(p, y, W),
}


def _strata(g: pd.DataFrame) -> np.ndarray | None:
    return g["fold"].to_numpy() if "fold" in g.columns else None


def _point_and_boot(g: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    p, y, s = g["p_fail"].to_numpy(float), g["fail"].to_numpy(float), _strata(g)
    iid = g["instance_id"].to_numpy()
    ones = np.ones((1, len(p)))
    out = {}
    boots = {m: [] for m in METRICS}
    for W in issue_bootstrap_weights(iid, n_boot, seed):
        for m, f in METRICS.items():
            boots[m].append(f(p, y, W, s, iid))
    for m, f in METRICS.items():
        out[m] = float(f(p, y, ones, s, iid)[0])
        b = np.concatenate(boots[m]) if boots[m] else np.array([np.nan])
        out[f"{m}_lo"], out[f"{m}_hi"] = _ci(b)
    out["auprc_fail"] = float(weighted_average_precision(p, y, ones)[0])
    return out


def add_cohorts(pred: pd.DataFrame, fixed_min_steps: int,
                seen_repos: set[str] | None) -> pd.DataFrame:
    parts = [pred.assign(cohort="alive"),
             pred[pred.n_steps >= fixed_min_steps].assign(cohort="fixed")]
    if seen_repos is not None:
        seen = pred["instance_id"].map(repo_of).isin(seen_repos)
        parts += [pred[seen].assign(cohort="seen_repo"), pred[~seen].assign(cohort="unseen_repo")]
    if "model_name" in pred.columns:
        parts += [g.assign(cohort=f"agent={m}") for m, g in pred.groupby("model_name")]
    return pd.concat(parts, ignore_index=True)


def repo_of(instance_id: str) -> str:
    return instance_id.rsplit("-", 1)[0]


def training_repos(split_file: Path) -> set[str]:
    tab = pd.read_csv(split_file)
    return set(tab.loc[tab.split.isin(["hmm_train", "clf_train"]), "instance_id"].map(repo_of))


def compute_metrics(pred: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                    fixed_min_steps: int = 31,
                    seen_repos: set[str] | None = None) -> pd.DataFrame:
    check_predictions(pred)
    rows = []
    for (method, cohort, t), g in add_cohorts(pred, fixed_min_steps, seen_repos).groupby(
            ["method", "cohort", "t"], sort=True):
        if g["fail"].nunique() < 2:
            continue
        rows.append({"method": method, "cohort": cohort, "t": t, "n_runs": len(g),
                     "n_issues": g["instance_id"].nunique(), "fail_rate": g["fail"].mean(),
                     "n_issues_mixed": int((g.groupby("instance_id")["fail"].nunique() == 2).sum()),
                     "brier_prior": g["fail"].mean() * (1 - g["fail"].mean()),
                     **_point_and_boot(g, n_boot, seed)})
    return pd.DataFrame(rows)


def paired_diff(pred: pd.DataFrame, method_a: str, method_b: str, n_boot: int = 1000,
                seed: int = 0, fixed_min_steps: int = 31,
                seen_repos: set[str] | None = None) -> pd.DataFrame:
    check_predictions(pred)
    sub = pred[pred.method.isin([method_a, method_b])]
    wide = sub.pivot_table(index=["run_id", "t"], columns="method", values="p_fail")
    if wide.isna().any().any():
        raise ValueError("methods were scored on different runs; cannot pair them")
    meta_cols = ["instance_id", "fail", "n_steps"] + \
        [c for c in ("model_name", "fold") if c in sub.columns]
    meta = sub.drop_duplicates(["run_id", "t"]).set_index(["run_id", "t"])[meta_cols]
    both = wide.join(meta).reset_index().assign(method="pair", p_fail=0.0)
    rows = []
    for (cohort, t), g in add_cohorts(both, fixed_min_steps, seen_repos).groupby(
            ["cohort", "t"], sort=True):
        if g["fail"].nunique() < 2:
            continue
        y, s, iid = g["fail"].to_numpy(float), _strata(g), g["instance_id"].to_numpy()
        pa, pb = g[method_a].to_numpy(float), g[method_b].to_numpy(float)
        ones = np.ones((1, len(y)))
        row = {"cohort": cohort, "t": t, "n_runs": len(g), "a": method_a, "b": method_b}
        boots = {m: [] for m in METRICS}
        for W in issue_bootstrap_weights(iid, n_boot, seed):
            for m, f in METRICS.items():
                boots[m].append(f(pb, y, W, s, iid) - f(pa, y, W, s, iid))
        for m, f in METRICS.items():
            b = np.concatenate(boots[m])
            row[f"d_{m}"] = float(f(pb, y, ones, s, iid)[0] - f(pa, y, ones, s, iid)[0])
            row[f"d_{m}_lo"], row[f"d_{m}_hi"] = _ci(b)
            row[f"p_b_better_{m}"] = float(np.mean(b < 0 if m == "brier" else b > 0))
        rows.append(row)
    return pd.DataFrame(rows)


def tradeoff(pred: pd.DataFrame, stop_fracs=(0.05, 0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    rows = []
    for (method, t), g in pred.groupby(["method", "t"], sort=True):
        fail = g["fail"].to_numpy(bool)
        remaining = (g["n_steps"] - t).to_numpy(float)
        p = g["p_fail"].to_numpy(float)
        strata = g["fold"].to_numpy() if "fold" in g.columns else np.zeros(len(g))
        for q in stop_fracs:
            stop = np.zeros(len(g), bool)
            for s_ in np.unique(strata):
                idx = np.flatnonzero(strata == s_)
                stop[idx[np.argsort(-p[idx], kind="stable")[:int(round(q * len(idx)))]]] = True
            k = int(stop.sum())
            rows.append({
                "method": method, "t": t, "stop_frac": q, "n_stopped": k,
                "fail_rate": fail.mean(),
                "precision_fail": fail[stop].mean() if k else np.nan,
                "failures_caught": (fail & stop).sum() / max(fail.sum(), 1),
                "successes_lost": (~fail & stop).sum() / max((~fail).sum(), 1),
                "compute_saved": remaining[stop].sum() / remaining.sum(),
            })
    return pd.DataFrame(rows)


ORACLE_METHOD = "oracle:issue_difficulty"


def issue_difficulty_oracle(pred: pd.DataFrame) -> pd.DataFrame:
    base = pred[pred["method"] != ORACLE_METHOD].drop_duplicates(["run_id", "t"]).copy()
    rate = base.drop_duplicates("run_id").groupby("instance_id")["fail"].mean()
    base["p_fail"] = base["instance_id"].map(rate).astype(float)
    base["method"] = ORACLE_METHOD
    return base


def check_predictions(pred: pd.DataFrame) -> None:
    missing = set(PRED_COLUMNS) - set(pred.columns)
    if missing:
        raise ValueError(f"predictions missing columns {sorted(missing)}")
    if (pred["n_steps"] <= pred["t"]).any():
        raise ValueError("predictions include runs with n_steps <= t (hard rule 3)")
    if pred.duplicated(["method", "run_id", "t"]).any():
        raise ValueError("duplicate (method, run_id, t) rows")
    if not pred["p_fail"].between(0, 1).all():
        raise ValueError("p_fail must be a probability in [0, 1]")


def format_metrics(metrics: pd.DataFrame, cohort: str = "alive",
                   metric: str = "auroc") -> pd.DataFrame:
    m = metrics[metrics.cohort == cohort].copy()
    m["cell"] = [f"{r[metric]:.3f} [{r[metric + '_lo']:.3f}, {r[metric + '_hi']:.3f}]"
                 for _, r in m.iterrows()]
    return m.pivot(index="t", columns="method", values="cell")


def print_report(metrics: pd.DataFrame, trade: pd.DataFrame | None = None,
                 paired: pd.DataFrame | None = None) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    alive = metrics[metrics.cohort == "alive"]
    print("\nCohort 'alive' (runs still going after t):")
    print(alive.groupby("t")[["n_runs", "n_issues", "fail_rate", "brier_prior"]].first().round(3))
    for metric, note in [("auroc", "higher is better; 0.5 = chance"),
                         ("auroc_wi", "WITHIN-ISSUE: failing vs successful runs of the same "
                                      "issue; issue difficulty factored out"),
                         ("auprc_success", "success = positive; chance = 1 - fail_rate"),
                         ("brier", "lower is better; compare with brier_prior above")]:
        print(f"\n{metric.upper()} [95% issue-bootstrap CI]  ({note})")
        print(format_metrics(metrics, "alive", metric))
    print("\nAUROC by cohort (point estimates):")
    print(metrics.pivot_table(index=["cohort", "t"], columns="method", values="auroc").round(3))
    if paired is not None and len(paired):
        print("\nPaired differences, b - a (same issue resamples):")
        for _, r in paired[paired.cohort == "alive"].iterrows():
            print(f"  t={r.t:3d}  {r.b} - {r.a}:  dAUROC={r.d_auroc:+.3f} "
                  f"[{r.d_auroc_lo:+.3f}, {r.d_auroc_hi:+.3f}]  P(b better)={r.p_b_better_auroc:.2f}"
                  + (f"   within-issue dAUROC={r.d_auroc_wi:+.3f} "
                     f"[{r.d_auroc_wi_lo:+.3f}, {r.d_auroc_wi_hi:+.3f}]" if "d_auroc_wi" in r else ""))
    if trade is not None and len(trade):
        t10 = trade[trade.t == (10 if (trade.t == 10).any() else trade.t.min())]
        print(f"\nStopping trade-off at t={t10.t.iloc[0]} (random stopping: precision = fail "
              "rate, successes lost = stop_frac):")
        print(t10.drop(columns=["t", "n_stopped"]).round(3).to_string(index=False))
