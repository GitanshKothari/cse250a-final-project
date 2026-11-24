from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import normalize_within_model

COMPARISONS = [
    ("hmm+behavior", "behavior"),
    ("hmm", "markov"),
    ("hmm+behavior", "markov+behavior"),
    ("hmm", "hmm_snapshot"),
    ("hmm", "behavior"),
    ("cc_hmm_raw", "cc_markov_raw"),
    ("cc_hmm", "cc_markov"),
    ("cc_hmm+behavior", "cc_markov+behavior"),
    ("cc_hmm+behavior", "behavior"),
    ("cc_hmm", "hmm"),
]


def evaluate_feature_sets(tab: pd.DataFrame, t: int, eval_split: str,
                          feature_sets: dict[str, list[str]], ec: dict,
                          raw_scores: dict[str, str] | None = None
                          ) -> tuple[list[dict], list[pd.DataFrame]]:
    tr, ev = tab[tab.split == "clf_train"], tab[tab.split == eval_split]
    if tr.target.nunique() < 2 or ev.target.nunique() < 2:
        print(f"t={t}: not enough runs of both classes, skipping")
        return [], []
    y_tr, y_ev = ~tr.target.to_numpy(), ~ev.target.to_numpy()
    cols_all = sorted(set(sum(feature_sets.values(), [])))
    trn, evn = normalize_within_model(tr, [ev], cols_all) if ec["normalize_within_model"] else (tr, ev)

    scores = {}
    for name, cols in feature_sets.items():
        clf = LogisticRegression(C=ec["C"], max_iter=5000, class_weight="balanced")
        clf.fit(trn[cols].to_numpy(), y_tr)
        scores[name] = clf.predict_proba(evn[cols].to_numpy())[:, 1]
    for name, col in (raw_scores or {}).items():
        scores[name] = ev[col].to_numpy()

    rows, preds = [], []
    for name, p in scores.items():
        rows.append({"t": t, "features": name, "n_train": len(tr), "n_eval": len(ev),
                     "fail_rate_eval": y_ev.mean(), "auroc": roc_auc_score(y_ev, p),
                     "auprc": average_precision_score(y_ev, p)})
        preds.append(pd.DataFrame({"t": t, "features": name, "run_id": ev.run_id.to_numpy(),
                                   "instance_id": ev.instance_id.to_numpy(),
                                   "failed": y_ev, "p_fail": p}))
        print(f"t={t:3d}  {name:<20} AUROC={rows[-1]['auroc']:.3f}  "
              f"AUPRC={rows[-1]['auprc']:.3f}  (n_eval={len(ev)}, fail rate={y_ev.mean():.2f})")
    return rows, preds


def concordant(f: np.ndarray, s: np.ndarray) -> float:
    s = np.sort(s)
    lo, hi = np.searchsorted(s, f, "left"), np.searchsorted(s, f, "right")
    return lo.sum() + 0.5 * (hi - lo).sum()


def pair_counts(P: np.ndarray, y: np.ndarray, group: np.ndarray, issue: np.ndarray):
    n_issue = issue.max() + 1
    C, N = np.zeros((n_issue, P.shape[1])), np.zeros(n_issue)
    for g in np.unique(group):
        m = group == g
        f, s = m & y, m & ~y
        if f.any() and s.any():
            i = issue[m][0]
            N[i] += f.sum() * s.sum()
            C[i] += [concordant(P[f, j], P[s, j]) for j in range(P.shape[1])]
    return C, N
