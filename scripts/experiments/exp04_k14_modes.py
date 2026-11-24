from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from hmm_agents.config import ROOT, load_config
from hmm_agents.hmm import CategoricalHMM
from hmm_agents.models.feature_lr import fit_or_load_hmm
from hmm_agents.parsing import SYMBOLS
from hmm_agents.state_match import (STICKY, common_symbols, distinctness, hellinger,
                                    match_states, run_occupancy)

OUT = ROOT / "results" / "experiments" / "exp04"
PD_MODEL = ROOT / "archive" / "parallel_dev" / "models_snapshot" / "hmm_K14_full.npz"
REPRO_H, REPRO_STAY = 0.17, 0.10
SIMILAR_H = 0.30
PD_MODES = {3: ("First edit", "editing"), 9: ("Clean edit", "editing"),
            8: ("Run once", "running"), 7: ("Wrap-up", "running"),
            5: ("Reading code", "exploring"), 12: ("Hit-or-miss searching", "exploring"),
            2: ("Check after a run", "exploring"), 6: ("Stuck searching", "loop"),
            4: ("Lost searching", "loop"), 13: ("Editing without progress", "loop"),
            1: ("Off-script commands", "loop"), 10: ("Broken-edit loop", "loop"),
            0: ("Run-and-search loop", "loop"), 11: ("Broken-command loop", "loop")}


def top_symbols(row: np.ndarray, n: int = 3) -> str:
    return ", ".join(f"{SYMBOLS[v]} {row[v]:.2f}" for v in np.argsort(-row)[:n])


def quality(h: float, s: float) -> str:
    if h < REPRO_H and s < REPRO_STAY:
        return "same"
    return "similar" if h < SIMILAR_H else "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--Ks", type=int, nargs="+", default=[8, 14])
    ap.add_argument("--n-eval", type=int, default=10000,
                    help="train runs (fixed random sample) used for Viterbi shares/occupancy")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    runs = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet",
                           columns=["instance_id", "split", "target", "n_steps", "symbols"])
    train = runs[runs.split.isin(["hmm_train", "clf_train"])].reset_index(drop=True)
    ev = train.sample(min(args.n_eval, len(train)), random_state=0).reset_index(drop=True)
    seqs = [np.asarray(s) for s in ev.symbols]
    fail = ~ev.target.to_numpy(bool)
    print(f"train runs {len(train):,}; evaluation sample {len(ev):,} "
          f"(fail rate {fail.mean():.3f})")

    pd_hmm = CategoricalHMM.load(PD_MODEL)
    pd_A, pd_B = pd_hmm.transmat_, pd_hmm.emissionprob_
    n_common = pd_B.shape[1]

    dist_rows, stab_rows, mode_rows, fits = [], [], [], {}
    for K in args.Ks:
        for seed in args.seeds:
            hmm = fit_or_load_hmm(train, {"K": K}, seed)
            fits[K, seed] = hmm
            d = distinctness(hmm, seqs)
            d.update(seed=seed, ll_per_step=float(hmm.score_samples(seqs).sum()
                                                  / sum(map(len, seqs))))
            dist_rows.append(d)
            print(f"K={K} seed={seed}: {d}")

            B_c, idle_mass = common_symbols(hmm.emissionprob_, n_common)
            H = hellinger(B_c, pd_B)
            S = np.abs(np.diag(hmm.transmat_)[:, None] - np.diag(pd_A)[None])
            occ = run_occupancy(hmm, seqs)
            f_occ, s_occ = occ[fail].mean(0), occ[~fail].mean(0)
            for k in range(K):
                j = int(np.argmin(H[k] + S[k]))
                mode_rows.append({
                    "K": K, "seed": seed, "state": k,
                    "emits": top_symbols(hmm.emissionprob_[k]),
                    "stay": hmm.transmat_[k, k], "sticky": hmm.transmat_[k, k] >= STICKY,
                    "idle_mass": idle_mass[k],
                    "pd_state": j, "pd_name": PD_MODES[j][0], "pd_family": PD_MODES[j][1],
                    "pd_stay": pd_A[j, j], "hellinger": H[k, j], "stay_diff": S[k, j],
                    "match": quality(H[k, j], S[k, j]),
                    "failed_occ": f_occ[k], "solved_occ": s_occ[k],
                    "failed_over_solved": f_occ[k] / s_occ[k] if s_occ[k] > 0 else np.inf})

        ref = fits[K, args.seeds[0]]
        for seed in args.seeds[1:]:
            m = fits[K, seed]
            r, c, h, s = match_states(ref.transmat_, ref.emissionprob_, m.transmat_,
                                      m.emissionprob_)
            for rk, k, hk, sk in zip(r, c, h, s):
                stab_rows.append({"K": K, "seed": seed, "ref_state": rk, "state": k,
                                  "ref_emits": top_symbols(ref.emissionprob_[rk]),
                                  "hellinger": hk, "stay_diff": sk,
                                  "reproduced": hk < REPRO_H and sk < REPRO_STAY})

    cover_rows = []
    for (K, seed), hmm in fits.items():
        if K != pd_B.shape[0]:
            continue
        r, c, h, s = match_states(pd_A, pd_B, hmm.transmat_, hmm.emissionprob_)
        for j, k, hj, sj in zip(r, c, h, s):
            cover_rows.append({"seed": seed, "pd_state": j, "pd_name": PD_MODES[j][0],
                               "pd_family": PD_MODES[j][1], "state": k, "hellinger": hj,
                               "stay_diff": sj, "match": quality(hj, sj)})

    dist, stab = pd.DataFrame(dist_rows), pd.DataFrame(stab_rows)
    modes, cover = pd.DataFrame(mode_rows), pd.DataFrame(cover_rows)
    dist.to_csv(OUT / "distinctness.csv", index=False)
    stab.to_csv(OUT / "stability.csv", index=False)
    modes.to_csv(OUT / "modes_vs_parallel_dev.csv", index=False)
    cover.to_csv(OUT / "pd_mode_coverage.csv", index=False)

    with open(OUT / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"exp04 (dev_docs/16, section 4). Train runs only; Viterbi statistics on a fixed "
                f"sample of {len(ev):,} train runs. Seeds {args.seeds}.\n\n")
        f.write("Distinctness (parallel_dev's K-selection checks):\n")
        f.write(dist[["K", "seed", "ll_per_step", "n_rare_modes", "smallest_mode_share",
                      "n_near_duplicate_pairs", "closest_pair", "closest_hellinger",
                      "closest_stay_diff", "n_sticky"]].round(4).to_string(index=False))
        if len(stab):
            f.write(f"\n\nStability vs seed {args.seeds[0]} (reproduced = Hellinger < "
                    f"{REPRO_H} and stay within {REPRO_STAY}):\n")
            f.write(stab.groupby(["K", "seed"]).reproduced.agg(["sum", "count"])
                    .rename(columns={"sum": "reproduced", "count": "states"}).to_string())
        if len(cover):
            f.write("\n\nparallel_dev's 14 named modes: one-to-one counterpart in each K=14 "
                    "fit (same / similar / none):\n")
            f.write(cover.pivot_table(index=["pd_family", "pd_name"], columns="seed",
                                      values="match", aggfunc="first").to_string())
        for (K, seed), g in modes.groupby(["K", "seed"]):
            f.write(f"\n\nK={K} seed={seed}: each state's nearest named mode\n")
            f.write(g.sort_values("failed_over_solved", ascending=False)[
                ["state", "emits", "stay", "idle_mass", "pd_name", "hellinger", "stay_diff",
                 "match", "failed_occ", "solved_occ", "failed_over_solved"]]
                .round(3).to_string(index=False))
        f.write("\n")
    print((OUT / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
