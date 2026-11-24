import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hmm_agents.config import load_config
from hmm_agents.features import CONTEXT_FEATURES, hmm_feature_names, normalize_within_model
from hmm_agents.hmm import CategoricalHMM

RARE = 0.01
DUP_EMIT = 0.15
DUP_STAY = 0.10

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
hc, ec = cfg["hmm"], cfg["early"]

df = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet")
val = df[df.split == "val"]
clf = df[df.split.isin(["clf_train", "val"])].reset_index(drop=True)
clf["log_issue_len"] = np.log1p(clf["issue_len"])
T_MAX = max(ec["prefixes"])
prefixes = [np.asarray(s[:T_MAX]) for s in clf.symbols]
sel = pd.read_csv(cfg["paths"]["results"] / "hmm_selection.csv").set_index("K")


def hellinger(p, q):
    return np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum())


rows = []
for K in hc["n_states_grid"]:
    m = CategoricalHMM.load(cfg["paths"]["models"] / f"hmm_K{K}.npz")
    A, B = m.transmat_, m.emissionprob_

    paths = m.viterbi([np.asarray(s) for s in val.symbols])
    share = np.bincount(np.concatenate(paths), minlength=K) / sum(map(len, paths))
    pairs = [(hellinger(B[i], B[j]), abs(A[i, i] - A[j, j]), i, j)
             for i in range(K) for j in range(i + 1, K)]
    dups = [p for p in pairs if p[0] < DUP_EMIT and p[1] < DUP_STAY]
    h, d, i, j = min(pairs)

    filt = m.filter(prefixes)
    aurocs = {}
    for t in ec["prefixes"]:
        alive = np.flatnonzero(clf.n_steps.to_numpy() > t)
        p_t = np.stack([filt[r][t - 1] for r in alive])
        ll = m.score_samples([prefixes[r][:t] for r in alive]) / t
        cols = hmm_feature_names(K) + CONTEXT_FEATURES
        tab = clf.iloc[alive][["model_name", "split", "target", "log_issue_len"]].reset_index(drop=True)
        tab[cols[:-1]] = np.column_stack([p_t, ll])
        tr, ev = tab[tab.split == "clf_train"], tab[tab.split == "val"]
        if ec["normalize_within_model"]:
            tr, ev = normalize_within_model(tr, [ev], cols)
        lr = LogisticRegression(C=ec["C"], max_iter=2000, class_weight="balanced")
        lr.fit(tr[cols].to_numpy(), ~tr.target.to_numpy())
        aurocs[f"auroc_t{t}"] = roc_auc_score(~ev.target.to_numpy(),
                                              lr.predict_proba(ev[cols].to_numpy())[:, 1])

    rows.append({"K": K, "val_ll_per_step": sel.loc[K, "val_ll_per_step"],
                 "n_rare_modes": int((share < RARE).sum()),
                 "smallest_mode_share": share.min(),
                 "n_near_duplicate_pairs": len(dups),
                 "closest_pair": f"S{i}-S{j}", "closest_hellinger": h,
                 "closest_stay_diff": d, **aurocs,
                 "auroc_mean": np.mean(list(aurocs.values()))})
    print({k: round(float(v), 4) if isinstance(v, float) else v for k, v in rows[-1].items()})

pd.DataFrame(rows).to_csv(cfg["paths"]["results"] / "k_comparison.csv", index=False)
