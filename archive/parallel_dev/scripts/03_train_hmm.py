import argparse
import json
import shutil

import numpy as np
import pandas as pd

from hmm_agents.config import load_config
from hmm_agents.hmm import CategoricalHMM
from hmm_agents.parsing import SYMBOLS
from hmm_agents.viz import plot_hmm

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--skip-grid", action="store_true",
                help="don't fit the grid; go straight to the chosen K (needs n_states)")
args = ap.parse_args()
cfg = load_config(args.config)
hc = cfg["hmm"]
models, results = cfg["paths"]["models"], cfg["paths"]["results"]
fixed_K = hc.get("n_states")

df = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet")
L = hc["max_len"] or 10**9
train_all = df[df.split == "hmm_train"]
train = train_all
if hc["max_train_runs"] and len(train) > hc["max_train_runs"]:
    train = train.sample(hc["max_train_runs"], random_state=cfg["seed"])
seqs_tr = [np.asarray(s[:L]) for s in train.symbols]
seqs_va = [np.asarray(s[:L]) for s in df[df.split == "val"].symbols]
n_tr, n_va = sum(map(len, seqs_tr)), sum(map(len, seqs_va))


def fit(K, seqs):
    m = CategoricalHMM(n_states=K, n_symbols=len(SYMBOLS), n_iter=hc["n_iter"], tol=hc["tol"],
                       n_init=hc["n_init"], pseudocount=hc["pseudocount"],
                       random_state=cfg["seed"])
    return m.fit(seqs)


def summary(m, seqs, n_steps):
    return {"K": m.n_states, "train_runs": len(seqs),
            "train_ll_per_step": m.score(seqs) / n_steps,
            "val_ll_per_step": m.score(seqs_va) / n_va, "bic": float(m.bic(seqs)),
            "n_params": m.n_free_params(), "em_iters": len(m.history_)}


def show(row):
    print({k: round(float(v), 4) if isinstance(v, float) else v for k, v in row.items()})


sel_path = results / "hmm_selection.csv"
if args.skip_grid:
    if fixed_K is None:
        ap.error("--skip-grid needs hmm.n_states in the config")
else:
    print(f"Grid: {len(seqs_tr):,} hmm_train runs ({n_tr:,} steps)   val runs: {len(seqs_va):,}")
    rows = []
    for K in hc["n_states_grid"]:
        m = fit(K, seqs_tr)
        m.save(models / f"hmm_K{K}.npz")
        rows.append(summary(m, seqs_tr, n_tr))
        show(rows[-1])
    sel = pd.DataFrame(rows)
    if sel_path.exists():
        old = pd.read_csv(sel_path)
        sel = pd.concat([old[~old.K.isin(sel.K)], sel]).sort_values("K")
    sel.to_csv(sel_path, index=False)

if fixed_K is not None:
    best_K, how = int(fixed_K), "fixed by hand (hmm.n_states)"
else:
    sel = pd.read_csv(sel_path)
    idx = sel.val_ll_per_step.idxmax() if hc["select_by"] == "val_loglik" else sel.bic.idxmin()
    best_K, how = int(sel.loc[idx, "K"]), f"by {hc['select_by']}"
print(f"\nChosen K = {best_K} ({how})")

if hc.get("refit_full"):
    seqs_full = [np.asarray(s[:L]) for s in train_all.symbols]
    n_full = sum(map(len, seqs_full))
    print(f"Refitting K={best_K} on all {len(seqs_full):,} hmm_train runs ({n_full:,} steps)...")
    final = fit(best_K, seqs_full)
    final_path = models / f"hmm_K{best_K}_full.npz"
    final.save(final_path)
    info = summary(final, seqs_full, n_full)
else:
    final_path = models / f"hmm_K{best_K}.npz"
    if not final_path.exists():
        print(f"hmm_K{best_K}.npz not found; fitting it on the grid sample")
        fit(best_K, seqs_tr).save(final_path)
    final = CategoricalHMM.load(final_path)
    info = summary(final, seqs_tr, n_tr)
show(info)
shutil.copy(final_path, models / "hmm_best.npz")
(results / "hmm_final.json").write_text(json.dumps(
    {**info, "chosen_by": how, "model_file": final_path.name}, indent=2))

plot_hmm(final, results / "figures")
