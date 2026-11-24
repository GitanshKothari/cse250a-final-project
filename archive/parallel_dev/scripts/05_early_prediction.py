import argparse

import numpy as np
import pandas as pd

from hmm_agents.config import load_config
from hmm_agents.features import (BEHAVIOR_FEATURES, CONTEXT_FEATURES, MARKOV_FEATURES,
                                 build_prefix_table, hmm_feature_names, hmm_memory_feature_names,
                                 sticky_states)
from hmm_agents.evaluate import evaluate_feature_sets
from hmm_agents.hmm import CategoricalHMM
from hmm_agents.markov import MarkovChain
from hmm_agents.parsing import SYMBOLS
from hmm_agents.viz import plot_auroc

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--eval-split", default="val", choices=["val"],
                help="use 'test' only once, for the final report")
args = ap.parse_args()
cfg = load_config(args.config)
ec, hc = cfg["early"], cfg["hmm"]
res = cfg["paths"]["results"]

hmm = CategoricalHMM.load(cfg["paths"]["models"] / "hmm_best.npz")
df_all = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet")
L = hc["max_len"] or 10**9
mc = MarkovChain(len(SYMBOLS), pseudocount=hc["pseudocount"]).fit(
    [np.asarray(s[:L]) for s in df_all[df_all.split == "hmm_train"].symbols])
mc.save(cfg["paths"]["models"] / "markov.npz")
print(f"HMM K={hmm.n_states}; sticky states (stay >= 0.9): "
      f"{', '.join(f'S{k}' for k in sticky_states(hmm))}")

df = df_all[df_all.split.isin(["clf_train", args.eval_split])].reset_index(drop=True)

HMM_NOW = hmm_feature_names(hmm.n_states)
HMM_MEM = hmm_memory_feature_names(hmm.n_states)
FEATURE_SETS = {
    "context_only": CONTEXT_FEATURES,
    "behavior": BEHAVIOR_FEATURES + CONTEXT_FEATURES,
    "markov": MARKOV_FEATURES + CONTEXT_FEATURES,
    "hmm_snapshot": HMM_NOW + CONTEXT_FEATURES,
    "hmm": HMM_NOW + HMM_MEM + CONTEXT_FEATURES,
    "markov+behavior": MARKOV_FEATURES + BEHAVIOR_FEATURES + CONTEXT_FEATURES,
    "hmm+behavior": HMM_NOW + HMM_MEM + BEHAVIOR_FEATURES + CONTEXT_FEATURES,
}
rows, preds = [], []
for t in ec["prefixes"]:
    tab = build_prefix_table(df, hmm, t, mc)
    r, p = evaluate_feature_sets(tab, t, args.eval_split, FEATURE_SETS, ec)
    rows += r
    preds += p

out = pd.DataFrame(rows)
out.to_csv(res / f"early_prediction_{args.eval_split}.csv", index=False)
pd.concat(preds).to_parquet(res / f"early_predictions_{args.eval_split}.parquet", index=False)
plot_auroc(out, res / "figures", args.eval_split)
print(f"\nNext: python scripts/05b_bootstrap_ci.py --eval-split {args.eval_split}")
