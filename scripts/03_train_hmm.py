import argparse
import shutil

import numpy as np
import pandas as pd

from hmm_agents.config import load_config
from hmm_agents.hmm import CategoricalHMM
from hmm_agents.parsing import SYMBOLS
from hmm_agents.viz import plot_hmm

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
hc = cfg["hmm"]

df = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet")
L = hc["max_len"] or 10**9
train = df[df.split == "hmm_train"]
if hc["max_train_runs"] and len(train) > hc["max_train_runs"]:
    train = train.sample(hc["max_train_runs"], random_state=cfg["seed"])
seqs_tr = [np.asarray(s[:L]) for s in train.symbols]
seqs_va = [np.asarray(s[:L]) for s in df[df.split == "val"].symbols]
n_tr, n_va = sum(map(len, seqs_tr)), sum(map(len, seqs_va))
print(f"HMM train runs: {len(seqs_tr):,} ({n_tr:,} steps)   val runs: {len(seqs_va):,}")

rows = []
for K in hc["n_states_grid"]:
    m = CategoricalHMM(n_states=K, n_symbols=len(SYMBOLS), n_iter=hc["n_iter"], tol=hc["tol"],
                       n_init=hc["n_init"], pseudocount=hc["pseudocount"],
                       random_state=cfg["seed"])
    m.fit(seqs_tr)
    rows.append({"K": K, "train_ll_per_step": m.score(seqs_tr) / n_tr,
                 "val_ll_per_step": m.score(seqs_va) / n_va, "bic": float(m.bic(seqs_tr)),
                 "n_params": m.n_free_params(), "em_iters": len(m.history_),
                 "restart_ll_best": max(m.restart_ll_), "restart_ll_worst": min(m.restart_ll_),
                 "restarts_near_best": sum(x > max(m.restart_ll_) - 0.005 for x in m.restart_ll_),
                 "n_restarts": len(m.restart_ll_)})
    m.save(cfg["paths"]["models"] / f"hmm_K{K}.npz")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})
    print(f"   restarts (train LL/step): {sorted(round(x, 4) for x in m.restart_ll_)[::-1]}")

sel = pd.DataFrame(rows)
sel.to_csv(cfg["paths"]["results"] / "hmm_selection.csv", index=False)
idx = sel.val_ll_per_step.idxmax() if hc["select_by"] == "val_loglik" else sel.bic.idxmin()
best_K = int(sel.loc[idx, "K"])
shutil.copy(cfg["paths"]["models"] / f"hmm_K{best_K}.npz",
            cfg["paths"]["models"] / "hmm_best.npz")
print(f"\nSelected K = {best_K} (by {hc['select_by']})")

best = CategoricalHMM.load(cfg["paths"]["models"] / "hmm_best.npz")
plot_hmm(best, cfg["paths"]["results"] / "figures")
