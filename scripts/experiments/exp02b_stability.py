from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from hmm_agents.config import ROOT, load_config
from hmm_agents.hmm import CategoricalHMM
from hmm_agents.models.feature_lr import hmm_cache_path

D = ROOT / "results" / "experiments" / "exp02"


def emissions(train: pd.DataFrame, seed: int) -> np.ndarray:
    path = hmm_cache_path(train, {"K": 8}, seed)
    if not path.exists():
        raise SystemExit(f"run exp02 --seed {seed} first ({path.name} not cached)")
    return CategoricalHMM.load(path).emissionprob_


def hellinger(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return np.sqrt(0.5 * ((np.sqrt(P)[:, None] - np.sqrt(Q)[None]) ** 2).sum(-1))


def main() -> None:
    seeds = sorted(int(p.name[4:]) for p in D.glob("seed*") if (p / "dwell.csv").exists())
    cfg = load_config()
    runs = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet", columns=["instance_id", "split"])
    train = runs[runs.split.isin(["hmm_train", "clf_train"])].reset_index(drop=True)
    ref_states = pd.read_csv(D / "seed0" / "states.csv")
    ref_B = emissions(train, 0)
    rows = []
    for s in seeds:
        tc = pd.read_csv(D / f"seed{s}" / "transition_contrast.csv")
        dw = pd.read_csv(D / f"seed{s}" / "dwell.csv")
        H = hellinger(ref_B, emissions(train, s))
        ref_idx, idx = linear_sum_assignment(H)
        back = dict(zip(idx, ref_idx))
        for r, k in zip(ref_idx, idx):
            out = tc[tc["from"] == k]
            self_ = out[out["to"] == k].iloc[0]
            moves = out[(out["to"] != k) & ((out.p_fail >= 0.01) | (out.p_success >= 0.01))]
            esc = moves.sort_values("log_ratio").iloc[0] if len(moves) else None
            d = dw[dw.state == k].iloc[0]
            rows.append({
                "seed": s, "ref_state": r, "ref_top": ref_states.top_symbols[r],
                "state": k, "hellinger": H[r, k],
                "self_fail": self_.p_fail, "self_success": self_.p_success,
                "escape_to_ref": back.get(int(esc["to"])) if esc is not None else None,
                "escape_fail": esc.p_fail if esc is not None else np.nan,
                "escape_success": esc.p_success if esc is not None else np.nan,
                "escape_ratio": np.exp(esc.log_ratio) if esc is not None else np.nan,
                "hazard1": d["hsmm_hazard@1"], "hazard5": d["hsmm_hazard@5"],
                "hazard10": d["hsmm_hazard@10"], "hmm_hazard": d["hmm_hazard (constant)"]})
    df = pd.DataFrame(rows).sort_values(["ref_state", "seed"])
    df.to_csv(D / "stability.csv", index=False)
    with open(D / "stability.txt", "w") as f:
        f.write(f"Seeds: {seeds}. States matched to seed 0 by emission (Hellinger; < 0.1 = "
                "near-identical, > 0.3 = a different state).\n\n")
        for r, g in df.groupby("ref_state"):
            f.write(f"S{r} (seed 0): {ref_states.top_symbols[r]}\n")
            f.write(g[["seed", "state", "hellinger", "self_fail", "self_success",
                       "escape_to_ref", "escape_ratio", "hazard1", "hazard5", "hazard10",
                       "hmm_hazard"]].round(3).to_string(index=False) + "\n\n")
    print((D / "stability.txt").read_text())


if __name__ == "__main__":
    main()
