import argparse

import numpy as np
import pandas as pd

from hmm_agents.config import load_config
from hmm_agents.hmm import CategoricalHMM
from hmm_agents.parsing import SYMBOLS
from hmm_agents.viz import plot_occupancy

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
res = cfg["paths"]["results"]

hmm = CategoricalHMM.load(cfg["paths"]["models"] / "hmm_best.npz")
K = hmm.n_states
df = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet")
df = df[df.split.isin(["clf_train", "val"])].reset_index(drop=True)
paths = hmm.viterbi([np.asarray(s) for s in df.symbols])

occ = np.stack([np.bincount(p, minlength=K) / len(p) for p in paths])
occ_df = pd.DataFrame(occ, columns=[f"state_{k}" for k in range(K)]).assign(target=df.target)
summary = occ_df.groupby("target").mean().T.rename(columns={True: "success", False: "failure"})
summary["top_symbols"] = [", ".join(SYMBOLS[v] for v in np.argsort(-hmm.emissionprob_[k])[:3])
                          for k in range(K)]
summary.to_csv(res / "state_occupancy.csv")
print("Mean fraction of steps in each state:\n", summary.round(3))


def trans_freq(ps):
    C = np.zeros((K, K))
    for p in ps:
        np.add.at(C, (p[:-1], p[1:]), 1)
    return C / C.sum(1, keepdims=True).clip(1)


T_s = trans_freq([p for p, y in zip(paths, df.target) if y])
T_f = trans_freq([p for p, y in zip(paths, df.target) if not y])
diff = pd.DataFrame(T_f - T_s, index=[f"from_{k}" for k in range(K)],
                    columns=[f"to_{k}" for k in range(K)])
diff.to_csv(res / "transition_diff.csv")
print("\nP(transition | failure) - P(transition | success):\n", diff.round(3))

rng = np.random.default_rng(cfg["seed"])
with open(res / "example_runs.txt", "w", encoding="utf-8") as f:
    for label in (True, False):
        for i in rng.choice(np.flatnonzero(df.target == label), 3, replace=False):
            r = df.iloc[i]
            f.write(f"=== {r.instance_id} | {r.model_name} | solved={r.target} | {r.n_steps} steps\n")
            for t, (s, cmd) in enumerate(zip(paths[i], r.commands)):
                f.write(f"  {t:3d}  S{s}  {SYMBOLS[r.symbols[t]]:<12}  {cmd[:90]}\n")
            f.write("\n")
print(f"\nWrote annotated runs to {res / 'example_runs.txt'}")
plot_occupancy(summary, res / "figures")
