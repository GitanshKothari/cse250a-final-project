import argparse

import numpy as np
import pandas as pd

from hmm_agents.config import load_config
from hmm_agents.evaluate import evaluate_feature_sets
from hmm_agents.features import (BEHAVIOR_FEATURES, CONTEXT_FEATURES, MARKOV_FEATURES,
                                 build_prefix_table, hmm_feature_names, hmm_memory_feature_names)
from hmm_agents.hmm import CategoricalHMM
from hmm_agents.markov import MarkovChain
from hmm_agents.parsing import SYMBOLS
from hmm_agents.viz import plot_auroc

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--eval-split", default="val", choices=["val"],
                help="use 'test' only once, for the final report")
ap.add_argument("--skip-fit", action="store_true", help="reuse the saved class models")
args = ap.parse_args()
cfg = load_config(args.config)
ec, hc, cc = cfg["early"], cfg["hmm"], cfg["class_conditional"]
models, res = cfg["paths"]["models"], cfg["paths"]["results"]
K = cc["n_states"]

df_all = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet")
L = hc["max_len"] or 10**9
fit_df = df_all[df_all.split == cc["fit_on"]]

class_models = {}
for cls, solved in (("failed", False), ("solved", True)):
    part = fit_df[fit_df.target == solved]
    cap = cc.get("max_train_runs_per_class")
    if cap and len(part) > cap:
        part = part.sample(cap, random_state=cfg["seed"])
    seqs = [np.asarray(s[:L]) for s in part.symbols]
    n = sum(map(len, seqs))
    hmm_path, mc_path = models / f"cc_hmm_{cls}.npz", models / f"cc_markov_{cls}.npz"
    if args.skip_fit:
        h, m = CategoricalHMM.load(hmm_path), MarkovChain.load(mc_path)
    else:
        print(f"Fitting {cls} models on {len(seqs):,} {cc['fit_on']} runs ({n:,} steps), K={K}...",
              flush=True)
        h = CategoricalHMM(n_states=K, n_symbols=len(SYMBOLS), n_iter=hc["n_iter"], tol=hc["tol"],
                           n_init=hc["n_init"], pseudocount=hc["pseudocount"],
                           random_state=cfg["seed"]).fit(seqs)
        m = MarkovChain(len(SYMBOLS), pseudocount=hc["pseudocount"]).fit(seqs)
        h.save(hmm_path)
        m.save(mc_path)
        print(f"  HMM LL/step {h.score(seqs) / n:.4f} ({len(h.history_)} EM iters)   "
              f"Markov LL/step {m.score(seqs) / n:.4f}", flush=True)
    class_models[cls] = (h, m)

hmm = CategoricalHMM.load(models / "hmm_best.npz")
mc = MarkovChain.load(models / "markov.npz")
df = df_all[df_all.split.isin(["clf_train", args.eval_split])].reset_index(drop=True)
(hf, mf), (hs, ms) = class_models["failed"], class_models["solved"]

HMM_ALL = hmm_feature_names(hmm.n_states) + hmm_memory_feature_names(hmm.n_states)
FEATURE_SETS = {
    "behavior": BEHAVIOR_FEATURES + CONTEXT_FEATURES,
    "hmm": HMM_ALL + CONTEXT_FEATURES,
    "markov+behavior": MARKOV_FEATURES + BEHAVIOR_FEATURES + CONTEXT_FEATURES,
    "hmm+behavior": HMM_ALL + BEHAVIOR_FEATURES + CONTEXT_FEATURES,
    "cc_markov": ["llr_markov"] + CONTEXT_FEATURES,
    "cc_hmm": ["llr_hmm"] + CONTEXT_FEATURES,
    "cc_markov+behavior": ["llr_markov"] + BEHAVIOR_FEATURES + CONTEXT_FEATURES,
    "cc_hmm+behavior": ["llr_hmm"] + BEHAVIOR_FEATURES + CONTEXT_FEATURES,
}
RAW = {"cc_markov_raw": "llr_markov", "cc_hmm_raw": "llr_hmm"}

rows, preds = [], []
for t in ec["prefixes"]:
    tab = build_prefix_table(df, hmm, t, mc)
    prefixes = [np.asarray(s[:t]) for s in df[df.n_steps > t].symbols]
    tab["llr_hmm"] = hf.score_samples(prefixes) - hs.score_samples(prefixes)
    tab["llr_markov"] = mf.score_samples(prefixes) - ms.score_samples(prefixes)
    r, p = evaluate_feature_sets(tab, t, args.eval_split, FEATURE_SETS, ec, RAW)
    rows += r
    preds += p

out = pd.DataFrame(rows)
out.to_csv(res / f"cc_prediction_{args.eval_split}.csv", index=False)
pd.concat(preds).to_parquet(res / f"cc_predictions_{args.eval_split}.parquet", index=False)
plot_auroc(out, res / "figures", f"{args.eval_split}_cc")
print(f"\nNext: python scripts/05b_bootstrap_ci.py --tag cc --eval-split {args.eval_split}")
