from __future__ import annotations

import datetime
import getpass
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import ROOT
from .evaluate import (compute_metrics, issue_difficulty_oracle, paired_diff, print_report,
                       tradeoff, training_repos, weighted_auroc)
from .features import clear_feature_cache
from .models import FailurePredictor, build_model, prefix_view
from .models.base import STEP_COLUMNS
from .splits import TRAIN_SPLITS, load_folds
from .viz import plot_auroc

MODES = ("cv", "val", "test")



def check_prefix_invariance(model: FailurePredictor, runs: pd.DataFrame, t: int,
                            n: int = 200, seed: int = 0) -> None:
    alive = runs[runs["n_steps"] > t]
    if len(alive) == 0:
        return
    sample = alive.sample(min(n, len(alive)), random_state=seed).reset_index(drop=True)
    base = model.predict_proba(prefix_view(sample, t), t)

    future = sample.copy()
    for c in STEP_COLUMNS:
        if c in future.columns:
            future[c] = [np.concatenate([np.asarray(v[:t]), np.asarray(v[t:])[::-1],
                                         np.asarray(v[:1])]) for v in future[c]]
    future["n_steps"] = future["n_steps"] + 1
    if "target" in future.columns:
        future["target"] = ~future["target"].astype(bool)
    if not np.allclose(base, model.predict_proba(prefix_view(future, t), t), atol=1e-9):
        raise AssertionError(f"{model.name}: prediction at t={t} depends on steps after t")

    perm = np.random.default_rng(seed).permutation(len(sample))
    shuffled = model.predict_proba(prefix_view(sample.iloc[perm], t), t)
    if not np.allclose(base[perm], shuffled, atol=1e-9):
        raise AssertionError(f"{model.name}: prediction depends on batch order")
    half = len(sample) // 2
    if half:
        part = model.predict_proba(prefix_view(sample.iloc[:half], t), t)
        if not np.allclose(base[:half], part, atol=1e-9):
            raise AssertionError(f"{model.name}: prediction depends on batch composition")



def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def provenance(cfg: dict) -> dict[str, Any]:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "runs_parquet_sha": _sha(cfg["paths"]["processed"] / "runs.parquet"),
        "splits_sha": _sha(ROOT / cfg["split"]["file"]),
        "folds_sha": _sha(ROOT / cfg["split"]["folds_file"]),
        "seed": cfg["seed"],
    }


def _log_access(mode: str, exp: dict, prov: dict, cfg: dict) -> None:
    logs = Path(cfg["paths"].get("logs", ROOT / "logs"))
    log = logs / ("val_runs.log" if mode == "val" else "test_access.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{prov['timestamp']}\t{getpass.getuser()}\texp={exp['name']}"
                f"\tcommit={prov['git_commit']}{'+dirty' if prov['git_dirty'] else ''}"
                f"\tclaim={exp.get('claim', '')!r}\n")


RUN_LOG = "experiment_runs.log"


def _log_run(exp: dict, mode: str, prov: dict, metrics: pd.DataFrame, cfg: dict,
             out: Path) -> None:
    t = cfg["eval"].get("primary_t", 10)
    m = metrics[(metrics["cohort"] == "alive") & (metrics["t"] == t)
                & ~metrics["method"].str.startswith("oracle:")]
    m = m.assign(base=m["method"].str.rsplit("@s", n=1).str[0])
    summ = m.groupby("base", sort=False)[["auroc", "auroc_wi"]].mean()
    cells = "; ".join(f"{b} {r.auroc:.3f}/{r.auroc_wi:.3f}" for b, r in summ.iterrows())
    logs = Path(cfg["paths"].get("logs", ROOT / "logs"))
    logs.mkdir(parents=True, exist_ok=True)
    kind = f"{mode}+rescore" if "rescored_at" in prov else mode
    with open(logs / RUN_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{getpass.getuser()}"
                f"\texp={exp.get('name')}\tmode={kind}"
                f"\tcommit={prov.get('git_commit', '?')}{'+dirty' if prov.get('git_dirty') else ''}"
                f"\tout={out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}"
                f"\tAUROC/within-issue @t={t}: {cells}\n")



MAX_GRID = 20


def _predict_rows(model, runs, t, method: str) -> pd.DataFrame:
    alive = runs[runs["n_steps"] > t].reset_index(drop=True)
    p = model.predict_proba(prefix_view(alive, t), t)
    return pd.DataFrame({"method": method, "run_id": alive["run_id"].to_numpy(),
                         "instance_id": alive["instance_id"].to_numpy(), "t": t,
                         "p_fail": np.asarray(p, float), "fail": ~alive["target"].to_numpy(bool),
                         "n_steps": alive["n_steps"].to_numpy(),
                         "model_name": alive["model_name"].to_numpy()})


def _fit(spec, train, prefixes, seed) -> FailurePredictor:
    model = build_model(spec, seed=seed)
    return model.fit(train.drop(columns=["target"]), ~train["target"].to_numpy(bool),
                     prefixes)


def expand_grid(grid: dict | None) -> list[dict]:
    combos = [{}]
    for key, values in (grid or {}).items():
        combos = [{**c, key: v} for c in combos for v in values]
    if len(combos) > MAX_GRID:
        raise ValueError(f"grid has {len(combos)} configs; the protocol allows {MAX_GRID}")
    return combos


def _method_name(name: str, seed: int, multi_seed: bool) -> str:
    return f"{name}@s{seed}" if multi_seed else name


def _cv_predictions(spec, train, folds, prefixes, seed, method) -> tuple[list, dict]:
    preds, config = [], None
    for k in sorted(folds.unique()):
        tr = train[folds != k].reset_index(drop=True)
        ho = train[folds == k].reset_index(drop=True)
        model = _fit(spec, tr, prefixes, seed)
        if k == folds.min():
            for t in prefixes:
                check_prefix_invariance(model, ho, t, seed=seed)
            config = model.get_config()
        preds += [_predict_rows(model, ho, t, method) for t in prefixes]
        print(f"   fold {int(k)}: fit on {len(tr):,} runs, scored {len(ho):,}")
    return preds, config


def _code_sha() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "src" / "hmm_agents").rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _cv_predictions_ckpt(spec, train, folds, prefixes, seed, method, ckpt_dir: Path,
                         data_key: str) -> tuple[list, dict]:
    key = hashlib.sha256(json.dumps({"spec": spec, "seed": seed, "prefixes": list(prefixes),
                                     "method": method, "data": data_key},
                                    sort_keys=True, default=str).encode()).hexdigest()[:16]
    stem = ckpt_dir / f"{method.replace('/', '_')}__{key}"
    pq, js = stem.with_suffix(".parquet"), stem.with_suffix(".json")
    if pq.exists() and js.exists():
        print(f"   [{method}] loaded checkpoint {pq.name}")
        return [pd.read_parquet(pq)], json.loads(js.read_text())
    preds, config = _cv_predictions(spec, train, folds, prefixes, seed, method)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(pq, index=False)
    js.write_text(json.dumps(config, default=str))
    return preds, config


def _primary_score(preds: list[pd.DataFrame], folds_map: pd.Series, t: int) -> float:
    p = pd.concat(preds, ignore_index=True)
    p = p[p["t"] == t]
    return float(weighted_auroc(p["p_fail"].to_numpy(float), p["fail"].to_numpy(float),
                                np.ones((1, len(p))),
                                p["instance_id"].map(folds_map).to_numpy())[0])


def _expand_pairs(pairs, multi: dict[str, bool], seeds: list[int]) -> list[tuple[str, str]]:
    out = []
    for a, b in pairs or []:
        for s in seeds:
            pair = (_method_name(a, s, multi.get(a, False)), _method_name(b, s, multi.get(b, False)))
            if pair not in out:
                out.append(pair)
    return out


def pairs_from_methods(pairs, methods: set[str]) -> list[tuple[str, str]]:
    seeds = sorted({int(m.rsplit("@s", 1)[1]) for m in methods
                    if "@s" in m and m.rsplit("@s", 1)[1].isdigit()})
    multi = {n: any(m.startswith(n + "@s") for m in methods) for p in pairs or [] for n in p}
    return [(a, b) for a, b in _expand_pairs(pairs, multi, seeds or [0])
            if {a, b} <= methods]


def run_experiment(exp: dict, cfg: dict, mode: str, final: bool = False,
                   n_boot: int | None = None) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "test" and not final:
        raise SystemExit("Refusing to score test without final=True (docs/PROTOCOL.md §6).")
    if mode in ("val", "test") and not exp.get("claim"):
        raise SystemExit("val/test runs need a pre-declared `claim` in the experiment file "
                         "(docs/PROTOCOL.md §4).")
    clear_feature_cache()
    prefixes = exp.get("prefixes", cfg["early"]["prefixes"])
    seeds = list(exp.get("seeds", [cfg["seed"]]))
    ev = cfg["eval"]
    primary_t = ev.get("primary_t", 10)
    n_boot = ev["n_boot"] if n_boot is None else n_boot

    runs = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet")
    train = runs[runs["split"].isin(TRAIN_SPLITS)].reset_index(drop=True)
    folds_map = load_folds(ROOT / cfg["split"]["folds_file"])
    prov = provenance(cfg)
    out = cfg["paths"]["results"] / exp["name"] / mode
    out.mkdir(parents=True, exist_ok=True)
    tuning_file = cfg["paths"]["results"] / exp["name"] / "cv" / "tuning.yaml"
    ckpt_dir = out / "checkpoints"
    data_key = f"{prov['runs_parquet_sha']}:{prov['folds_sha']}:{_code_sha()}"
    chosen: dict[str, dict] = {}
    if mode != "cv" and any("grid" in m for m in exp["models"]):
        if not tuning_file.exists():
            raise SystemExit(f"{tuning_file} missing: run --mode cv first to tune the grid.")
        chosen = yaml.safe_load(tuning_file.read_text()) or {}
    if mode in ("val", "test"):
        _log_access(mode, exp, prov, cfg)

    preds, coefs, configs, multi, tuning_rows = [], [], {}, {}, []
    for spec in exp["models"]:
        name = spec["name"]
        base = {k: v for k, v in spec.items() if k != "grid"}
        grid = spec.get("grid")
        stochastic = build_model(base, seed=seeds[0]).stochastic
        model_seeds = seeds if stochastic else seeds[:1]
        multi[name] = len(model_seeds) > 1
        print(f"\n== {name} ({mode}; {'seeds ' + str(model_seeds) if multi[name] else 'seed ' + str(model_seeds[0])})")

        if mode == "cv":
            folds = train["instance_id"].map(folds_map)
            if folds.isna().any():
                raise ValueError("train issues missing from folds file")
            if grid:
                scored = []
                for params in expand_grid(grid):
                    print(f"   config {params}")
                    p_, c_ = _cv_predictions_ckpt({**base, **params}, train, folds, prefixes,
                                                  model_seeds[0],
                                                  _method_name(name, model_seeds[0], multi[name]),
                                                  ckpt_dir, data_key)
                    score = _primary_score(p_, folds_map, primary_t)
                    scored.append((score, params, p_, c_))
                    tuning_rows.append({"model": name, **{f"param_{k}": v for k, v in params.items()},
                                        f"cv_auroc_t{primary_t}": score})
                    print(f"     within-fold AUROC at t={primary_t}: {score:.4f}")
                best = max(scored, key=lambda r: r[0])
                chosen[name] = best[1]
                base = {**base, **best[1]}
                preds += best[2]
                configs[_method_name(name, model_seeds[0], multi[name])] = best[3]
                rest = model_seeds[1:]
                print(f"   chosen: {best[1]}")
            else:
                rest = model_seeds
            for s in rest:
                m = _method_name(name, s, multi[name])
                p_, c_ = _cv_predictions_ckpt(base, train, folds, prefixes, s, m, ckpt_dir,
                                              data_key)
                preds += p_
                configs[m] = c_
        else:
            if grid:
                if name not in chosen:
                    raise SystemExit(f"no tuned config for {name!r} in {tuning_file}")
                base = {**base, **chosen[name]}
                print(f"   using CV-tuned config {chosen[name]}")
            held = runs[runs["split"] == mode].reset_index(drop=True)
            (out / "models").mkdir(exist_ok=True)
            for s in model_seeds:
                m = _method_name(name, s, multi[name])
                model = _fit(base, train, prefixes, s)
                for t in prefixes:
                    check_prefix_invariance(model, held, t, seed=s)
                preds += [_predict_rows(model, held, t, m) for t in prefixes]
                c = model.coefficients() if hasattr(model, "coefficients") else None
                if c is not None:
                    coefs.append(c.assign(method=m))
                model.save(out / "models" / f"{m}.pkl")
                configs[m] = model.get_config()
                print(f"   [{m}] fit on {len(train):,} train runs, scored {len(held):,} {mode} "
                      f"runs; saved models/{m}.pkl")
        print("   prefix-invariance check passed")

    if mode == "cv" and chosen:
        tuning_file.write_text(yaml.safe_dump(chosen, sort_keys=False))
        pd.DataFrame(tuning_rows).to_csv(out / "tuning.csv", index=False)
        print(f"\nTuning choices -> {tuning_file}")

    pred = pd.concat(preds, ignore_index=True)
    if mode == "cv":
        pred["fold"] = pred["instance_id"].map(folds_map)
    pred.to_parquet(out / "predictions.parquet", index=False)
    if ckpt_dir.exists():
        done = out / "all_fits"
        if done.exists():
            for f in done.iterdir():
                f.unlink()
            done.rmdir()
        ckpt_dir.rename(done)
    if coefs:
        pd.concat(coefs, ignore_index=True).to_csv(out / "coefficients.csv", index=False)
    exp_run = {**exp, "_expanded_pairs": _expand_pairs(exp.get("paired"), multi, seeds)}
    score_predictions(pred, exp_run, cfg, mode, out, n_boot, prov, configs, prefixes)
    return out


def score_predictions(pred, exp, cfg, mode, out, n_boot, prov=None, configs=None,
                      prefixes=None) -> None:
    seed, ev = cfg["seed"], cfg["eval"]

    seen = None if mode == "cv" else training_repos(ROOT / cfg["split"]["file"])
    print(f"\nBootstrapping {n_boot} issue resamples ...")
    metrics = compute_metrics(pd.concat([pred, issue_difficulty_oracle(pred)], ignore_index=True),
                              n_boot, seed, ev["fixed_min_steps"], seen)
    trade = tradeoff(pred, ev["stop_fracs"])
    pairs = exp.get("_expanded_pairs")
    if pairs is None:
        pairs = pairs_from_methods(exp.get("paired", []), set(pred["method"]))
    pairs = [(a, b) for a, b in pairs if {a, b} <= set(pred["method"])]
    paired = pd.concat([paired_diff(pred, a, b, n_boot, seed, ev["fixed_min_steps"], seen)
                        for a, b in pairs], ignore_index=True) if pairs else pd.DataFrame()
    metrics.to_csv(out / "metrics.csv", index=False)
    trade.to_csv(out / "tradeoff.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    summary = seed_summary(metrics, paired)
    if summary is not None:
        summary.to_csv(out / "seed_summary.csv", index=False)
    exp_clean = {k: v for k, v in exp.items() if not k.startswith("_")}
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"experiment": exp_clean, "mode": mode, "provenance": prov,
                        "models": configs, "prefixes": prefixes, "n_boot": n_boot,
                        "leakage_check": "passed"}, f, sort_keys=False)
    _log_run(exp_clean, mode, prov or {}, metrics, cfg, out)
    print_report(metrics, trade, paired)
    if summary is not None:
        print("\nAcross seeds (alive cohort; mean [min, max]):")
        alive = summary[summary.cohort == "alive"]
        for _, r in alive.iterrows():
            print(f"  t={int(r['t']):3d}  {r['kind']:<8} {r['item']:<40} {r['mean']:+.3f} "
                  f"[{r['min']:+.3f}, {r['max']:+.3f}]  (n_seeds={int(r['n_seeds'])})")
    plot_auroc(metrics, out, mode)


def seed_summary(metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame | None:
    rows = []
    m = metrics[metrics["method"].str.contains("@s", regex=False)]
    if len(m):
        m = m.assign(item=m["method"].str.rsplit("@s", n=1).str[0])
        g = m.groupby(["item", "cohort", "t"])["auroc"]
        rows.append(g.agg(["mean", "std", "min", "max", "size"]).reset_index()
                    .rename(columns={"size": "n_seeds"}).assign(kind="auroc"))
    if len(paired):
        pp = paired[paired["a"].str.contains("@s", regex=False)
                    | paired["b"].str.contains("@s", regex=False)]
        if len(pp):
            strip = lambda x: x.str.rsplit("@s", n=1).str[0]
            pp = pp.assign(item=strip(pp["b"]) + " - " + strip(pp["a"]))
            g = pp.groupby(["item", "cohort", "t"])["d_auroc"]
            rows.append(g.agg(["mean", "std", "min", "max", "size"]).reset_index()
                        .rename(columns={"size": "n_seeds"}).assign(kind="d_auroc"))
    return pd.concat(rows, ignore_index=True) if rows else None


def rescore(exp: dict, cfg: dict, mode: str, n_boot: int | None = None) -> Path:
    out = cfg["paths"]["results"] / exp["name"] / mode
    pred = pd.read_parquet(out / "predictions.parquet")
    if mode == "cv":
        pred["fold"] = pred["instance_id"].map(load_folds(ROOT / cfg["split"]["folds_file"]))
    if (out / "config.yaml").exists():
        with open(out / "config.yaml", encoding="utf-8") as f:
            old = yaml.safe_load(f)
    else:
        old = {"provenance": {"note": "original config.yaml missing (scoring was interrupted); "
                              "fit provenance unknown", "rescored_from_current_config": True},
               "models": None,
               "prefixes": sorted(int(t) for t in pred["t"].unique())}
    n_boot = cfg["eval"]["n_boot"] if n_boot is None else n_boot
    prov = {**old["provenance"], "rescored_at": provenance(cfg)["timestamp"],
            "rescored_commit": provenance(cfg)["git_commit"]}
    score_predictions(pred, exp, cfg, mode, out, n_boot, prov, old["models"], old["prefixes"])
    return out
