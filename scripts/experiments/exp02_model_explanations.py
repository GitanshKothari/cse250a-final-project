from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from hmm_agents.config import ROOT, load_config
from hmm_agents.models.feature_lr import fit_or_load_hmm
from hmm_agents.models.generative import ClassHMMScorer
from hmm_agents.parsing import SYMBOLS

OUT = ROOT / "results" / "experiments" / "exp02"


def state_names(B: np.ndarray, top: int = 3) -> list[str]:
    return [", ".join(f"{SYMBOLS[v]} {B[k, v]:.2f}" for v in np.argsort(-B[k])[:top])
            for k in range(B.shape[0])]


def geometric_dwell(a_kk: float, max_d: int) -> np.ndarray:
    d = np.arange(1, max_d + 1)
    return (1 - a_kk) * a_kk ** (d - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed
    out = OUT / f"seed{seed}"
    cfg = load_config()
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet")
    train = runs[runs.split.isin(["hmm_train", "clf_train"])].reset_index(drop=True)
    y_fail = ~train["target"].to_numpy(bool)
    print(f"seed {seed}; train runs: {len(train):,} (fail rate {y_fail.mean():.3f})")

    base = fit_or_load_hmm(train, {"K": 8}, seed)
    K = base.n_states
    names = state_names(base.emissionprob_)
    occ = np.concatenate(base.filter([np.asarray(s[:150]) for s in train.symbols.iloc[:20000]])).mean(0)
    pd.DataFrame({"state": range(K), "top_symbols": names, "mean_filtered_occupancy": occ}) \
        .to_csv(out / "states.csv", index=False)

    cc = ClassHMMScorer(n_iter=30).fit(train, y_fail, [10], seed, base)
    C = cc.transition_contrast()
    rows = [{"from": i, "to": j, "from_state": names[i], "to_state": names[j],
             "p_fail": cc.A_[0][i, j], "p_success": cc.A_[1][i, j], "log_ratio": C[i, j]}
            for i in range(K) for j in range(K)]
    tc = pd.DataFrame(rows).sort_values("log_ratio", ascending=False)
    tc.to_csv(out / "transition_contrast.csv", index=False)

    hs = fit_or_load_hmm(train, {"K": 8, "n_phases": 3}, seed)
    max_d = 200
    pmf = hs.dwell_pmf(max_d)
    drows = []
    for k in range(K):
        g = geometric_dwell(base.transmat_[k, k], max_d)
        p = pmf[k]
        surv = 1 - np.cumsum(p) + p
        hazard = p / np.maximum(surv, 1e-12)
        drows.append({"state": k, "top_symbols": names[k],
                      "hmm_mean": (np.arange(1, max_d + 1) * g).sum(),
                      "hsmm_mean": (np.arange(1, max_d + 1) * p).sum(),
                      "hmm_P(d>=5)": g[4:].sum(), "hsmm_P(d>=5)": p[4:].sum(),
                      "hmm_P(d>=10)": g[9:].sum(), "hsmm_P(d>=10)": p[9:].sum(),
                      "hsmm_hazard@1": hazard[0], "hsmm_hazard@3": hazard[2],
                      "hsmm_hazard@5": hazard[4], "hsmm_hazard@10": hazard[9],
                      "hmm_hazard (constant)": 1 - base.transmat_[k, k]})
    dw = pd.DataFrame(drows)
    dw.to_csv(out / "dwell.csv", index=False)

    with open(out / "summary.txt", "w") as f:
        f.write("States of the plain HMM (top emissions) and mean filtered occupancy:\n")
        for k in range(K):
            f.write(f"  S{k}: {names[k]}   (occupancy {occ[k]:.3f})\n")
        f.write(f"\nClass-conditional HMM ({len(cc.history_)} EM iterations). Transitions that "
                "failing runs make MORE often than successful runs (log ratio > 0), among "
                "moves with p >= 0.02 in either class:\n")
        big = tc[(tc.p_fail >= 0.02) | (tc.p_success >= 0.02)]
        for _, r in pd.concat([big.head(8), big.tail(8)]).iterrows():
            f.write(f"  S{r['from']} -> S{r['to']}: fail {r.p_fail:.3f} vs success "
                    f"{r.p_success:.3f}  (x{np.exp(r.log_ratio):.2f})\n")
        f.write("\nDwell times: plain HMM (geometric) vs explicit-duration HMM:\n")
        f.write(dw.round(3).to_string(index=False) + "\n")
    print((out / "summary.txt").read_text())


if __name__ == "__main__":
    main()
