from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import yaml

from hmm_agents.config import ROOT, load_config
from hmm_agents.models.feature_lr import (HMM_DEFAULTS, HMM_GROUPS, HMM_VARIANT_DEFAULTS,
                                          LDS_DEFAULTS, fit_or_load_hmm, fit_or_load_lds,
                                          hmm_cache_path, lds_cache_path)
from hmm_agents.runner import expand_grid
from hmm_agents.splits import TRAIN_SPLITS, load_folds


def _cost(kind: str, s: dict) -> float:
    if kind == "lds":
        return 2.0
    n = s["K"] * s.get("n_phases", 1)
    restarts = s.get("phase_n_init", 1) if s.get("n_phases", 1) > 1 else s["n_init"]
    return n * n * restarts * (1.5 if s.get("alphabet") == "rich" else 1.0)


def jobs_for(exp: dict, model: str | None, params: dict | None) -> list[tuple[str, dict]]:
    out = []
    for spec in exp["models"]:
        if model and spec["name"] != model:
            continue
        configs = [params] if params is not None else expand_grid(spec.get("grid"))
        for cfg in configs:
            s = {**{k: v for k, v in spec.items() if k != "grid"}, **cfg}
            groups = set(s.get("groups", []))
            if s.get("hmm_data", "split") != "all":
                continue
            if (HMM_GROUPS | {"cchmm"}) & groups:
                hs = {**HMM_DEFAULTS, **HMM_VARIANT_DEFAULTS, **s.get("hmm", {})}
                if hs["n_phases"] > 1:
                    base = {k: v for k, v in hs.items() if k not in ("n_phases", "phase_n_init")}
                    base.update({"n_phases": 1, "phase_n_init": 1})
                    out.append(("hmm", base))
                out.append(("hmm", hs))
            if "lds" in groups:
                out.append(("lds", {**LDS_DEFAULTS, **s.get("lds", {})}))
    uniq = []
    for j in out:
        if j not in uniq:
            uniq.append(j)
    return sorted(uniq, key=lambda j: (j[1].get("n_phases", 1) > 1, -_cost(*j)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=str)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--model")
    ap.add_argument("--params", type=json.loads)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    exp = yaml.safe_load(open(a.exp))
    cfg = load_config()
    jobs = [(kind, s, seed) for seed in a.seeds for kind, s in jobs_for(exp, a.model, a.params)]
    mine = jobs if a.list else jobs[a.worker::a.n_workers]
    cols = ["instance_id", "split", "symbols"]
    if any(k == "lds" or s.get("alphabet") == "rich" for k, s, _ in mine):
        cols += ["no_match", "success_msg", "test_passed", "commands", "viewed_files",
                 "edited_files"]
    runs = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet", columns=cols)
    train = runs[runs["split"].isin(TRAIN_SPLITS)].reset_index(drop=True)
    folds = train["instance_id"].map(load_folds(ROOT / cfg["split"]["folds_file"]))
    tr = train[folds != a.fold].reset_index(drop=True)

    for kind, s, seed in mine:
        path = (lds_cache_path if kind == "lds" else hmm_cache_path)(tr, s, seed)
        if a.list:
            print(f"{kind} {s} seed={seed} cached={path.exists()}")
            continue
        if path.exists():
            continue
        t0 = time.time()
        if kind == "hmm" and s.get("n_phases", 1) > 1:
            base = {k: v for k, v in s.items() if k not in ("n_phases", "phase_n_init")}
            base_path = hmm_cache_path(tr, {**base, "n_phases": 1, "phase_n_init": 1}, seed)
            while not base_path.exists() and time.time() - t0 < 4 * 3600:
                time.sleep(30)
        if kind == "lds":
            fit_or_load_lds(tr, s, seed)
        else:
            fit_or_load_hmm(tr, s, seed)
        short = {k: v for k, v in s.items()
                 if k in ("K", "alphabet", "n_phases", "d")}
        print(f"fold {a.fold} {kind} {short} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    if not a.list:
        print(f"fold {a.fold} worker {a.worker} done", flush=True)


if __name__ == "__main__":
    main()
