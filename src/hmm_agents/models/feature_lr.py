from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ..config import ROOT
from ..features import (BEHAVIOR_FEATURES, CONTEXT_FEATURES, behavior_table,
                        hmm_feature_names, hmm_history_features, hmm_history_names,
                        ngram_features, ngram_names, trigram_matrix, trigram_name)
import pickle

from ..encoding import N_RICH, encode_prefixes, encode_run, rich_symbols
from ..lds import GaussianLDS
from ..hmm import CategoricalHMM, PhaseHMM
from ..parsing import SYMBOLS
from .base import FailurePredictor, prefix_view
from .generative import SCORERS

HMM_GROUPS = {"hmm", "hmm_hist"}
SUPERVISED_GROUPS = set(SCORERS)
HGB_DEFAULTS = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31,
                "min_samples_leaf": 50, "l2_regularization": 1.0, "early_stopping": False}
HMM_DEFAULTS = {"K": 8, "n_init": 10, "n_iter": 100, "tol": 1e-4, "pseudocount": 0.01,
                "max_train_runs": 8000, "max_len": 150, "path": None}
LDS_DEFAULTS = {"d": 4, "n_iter": 30, "tol": 1e-4, "max_train_runs": 8000, "max_len": 150}
HMM_VARIANT_DEFAULTS = {"alphabet": "base", "n_phases": 1, "phase_n_init": 1}


def hmm_symbols(runs: pd.DataFrame, alphabet: str, t: int | None = None) -> list[np.ndarray]:
    if alphabet == "rich":
        return [rich_symbols(r, t) for r in runs.to_dict("records")]
    return [np.asarray(s if t is None else s[:t]) for s in runs["symbols"]]


class WithinModelScaler:
    def fit(self, X: pd.DataFrame, model_name: pd.Series) -> "WithinModelScaler":
        X = X.astype(float)
        self.g_mu = X.mean()
        self.g_sd = X.std().replace(0, 1).fillna(1)
        g = X.groupby(model_name.to_numpy())
        self.mu, self.sd = g.mean(), g.std()
        return self

    def transform(self, X: pd.DataFrame, model_name: pd.Series) -> np.ndarray:
        X = X.astype(float).copy()
        names = model_name.to_numpy()
        for m in np.unique(names):
            idx = names == m
            if m in self.mu.index:
                mu, sd = self.mu.loc[m], self.sd.loc[m].replace(0, 1).fillna(1)
            else:
                mu, sd = self.g_mu, self.g_sd
            X.loc[idx] = ((X.loc[idx] - mu) / sd).to_numpy()
        return X.to_numpy()


def _hmm_cache_key(issues: list[str], settings: dict, seed: int) -> str:
    blob = json.dumps({"issues": sorted(issues), "settings": settings, "seed": seed,
                       "n_symbols": len(SYMBOLS)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def hmm_cache_path(runs: pd.DataFrame, settings: dict, seed: int,
                   cache_dir: Path | None = None) -> Path:
    s = {**HMM_DEFAULTS, **HMM_VARIANT_DEFAULTS, **settings}
    fit_settings = {k: v for k, v in s.items()
                    if k != "path" and not (k in HMM_VARIANT_DEFAULTS and v == HMM_VARIANT_DEFAULTS[k])}
    key = _hmm_cache_key(list(runs["instance_id"].unique()), fit_settings, seed)
    return (cache_dir or ROOT / "models" / "cache") / f"hmm_{key}.npz"


def lds_cache_path(runs: pd.DataFrame, settings: dict, seed: int,
                   cache_dir: Path | None = None) -> Path:
    s = {**LDS_DEFAULTS, **settings}
    key = _hmm_cache_key(list(runs["instance_id"].unique()), {"lds": s}, seed)
    return (cache_dir or ROOT / "models" / "cache") / f"lds_{key}.pkl"


def fit_or_load_hmm(runs: pd.DataFrame, settings: dict, seed: int,
                    cache_dir: Path | None = None) -> CategoricalHMM:
    s = {**HMM_DEFAULTS, **HMM_VARIANT_DEFAULTS, **settings}
    cls = PhaseHMM if s["n_phases"] > 1 else CategoricalHMM
    if s.get("path"):
        return cls.load(ROOT / s["path"])
    cache_dir = cache_dir or ROOT / "models" / "cache"
    path = hmm_cache_path(runs, settings, seed, cache_dir)
    if path.exists():
        return cls.load(path)
    train = runs
    if s["max_train_runs"] and len(train) > s["max_train_runs"]:
        train = train.sample(s["max_train_runs"], random_state=seed)
    L = s["max_len"] or 10**9
    n_sym = N_RICH if s["alphabet"] == "rich" else len(SYMBOLS)
    kw = dict(n_symbols=n_sym, n_iter=s["n_iter"], tol=s["tol"],
              n_init=s["phase_n_init"] if s["n_phases"] > 1 else s["n_init"],
              pseudocount=s["pseudocount"], random_state=seed)
    hmm = PhaseHMM(s["K"], s["n_phases"], **kw) if s["n_phases"] > 1 else \
        CategoricalHMM(n_states=s["K"], **kw)
    if s["n_phases"] > 1:
        base_settings = {k: v for k, v in settings.items() if k not in ("n_phases", "phase_n_init")}
        base = fit_or_load_hmm(runs, base_settings, seed, cache_dir)
        hmm.init_params = PhaseHMM.params_from_hmm(base, s["n_phases"])
    hmm.fit([x[:L] for x in hmm_symbols(train, s["alphabet"])])
    cache_dir.mkdir(parents=True, exist_ok=True)
    hmm.save(path)
    return hmm


def fit_or_load_lds(runs: pd.DataFrame, settings: dict, seed: int,
                    cache_dir: Path | None = None) -> GaussianLDS:
    s = {**LDS_DEFAULTS, **settings}
    cache_dir = cache_dir or ROOT / "models" / "cache"
    path = lds_cache_path(runs, settings, seed, cache_dir)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    train = runs
    if s["max_train_runs"] and len(train) > s["max_train_runs"]:
        train = train.sample(s["max_train_runs"], random_state=seed)
    seqs = [encode_run(r, s["max_len"]) for r in train.to_dict("records")]
    lds = GaussianLDS(d=s["d"], n_iter=s["n_iter"], tol=s["tol"], random_state=seed).fit(seqs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(lds, f)
    return lds


def lds_names(d: int) -> list[str]:
    return [f"lds_m{i}" for i in range(d)] + [f"lds_avg{i}" for i in range(d)] + ["lds_ll"]


class FeatureLR(FailurePredictor):
    def __init__(self, name: str, groups: list[str], C: float = 1.0,
                 class_weight: str | None = None, normalize_within_model: bool = True,
                 fit_on: str = "all", hmm: dict | None = None,
                 trigram_min_frac: float = 0.01, hmm_data: str = "split",
                 classifier: str = "lr", clf_params: dict | None = None,
                 landmark: bool = False, scorers: dict | None = None,
                 crossfit_folds: int = 5, lds: dict | None = None, seed: int = 0):
        self.name, self.groups, self.C, self.class_weight = name, list(groups), C, class_weight
        self.normalize, self.fit_on, self.seed = normalize_within_model, fit_on, seed
        self.trigram_min_frac = trigram_min_frac
        if hmm_data not in ("split", "all"):
            raise ValueError(f"hmm_data must be 'split' or 'all', got {hmm_data!r}")
        self.hmm_data = hmm_data
        if classifier not in ("lr", "hgb"):
            raise ValueError(f"classifier must be 'lr' or 'hgb', got {classifier!r}")
        self.classifier, self.clf_params, self.landmark = classifier, dict(clf_params or {}), landmark
        if landmark and "trigram" in groups:
            raise ValueError("landmark models need the same columns at every t (no trigram)")
        self.tri_cols_: dict[int, np.ndarray] = {}
        self.supervised = [g for g in self.groups if g in SUPERVISED_GROUPS]
        self.scorer_specs = {g: dict((scorers or {}).get(g, {})) for g in self.supervised}
        self.crossfit_folds = crossfit_folds
        self.needs_hmm = bool((HMM_GROUPS | {"cchmm"}) & set(self.groups))
        self.needs_lds = "lds" in self.groups
        self.lds_settings = {**LDS_DEFAULTS, **(lds or {})} if self.needs_lds else None
        self.stochastic = self.needs_hmm or self.needs_lds
        self.hmm_settings = ({**HMM_DEFAULTS, **HMM_VARIANT_DEFAULTS, **(hmm or {})}
                             if self.needs_hmm else None)
        if "cchmm" in self.groups and (self.hmm_settings["alphabet"] != "base"
                                       or self.hmm_settings["n_phases"] != 1):
            raise ValueError("cchmm is initialised from a plain 12-symbol HMM")
        unknown = set(self.groups) - {"context", "behavior", "ngram", "trigram", "lds",
                                      *HMM_GROUPS, *SUPERVISED_GROUPS}
        if unknown:
            raise ValueError(f"unknown feature groups {unknown}")

    def _K(self) -> int:
        return getattr(self.hmm_, "n_macro", self.hmm_.n_states) if self.needs_hmm else 0

    def columns(self, t: int | None = None) -> list[str]:
        K = self._K()
        names = {"context": CONTEXT_FEATURES, "behavior": BEHAVIOR_FEATURES,
                 "hmm": hmm_feature_names(K) if K else [],
                 "hmm_hist": hmm_history_names(K) if K else [], "ngram": ngram_names(),
                 "trigram": [trigram_name(i) for i in self.tri_cols_.get(t, [])],
                 "lds": lds_names(self.lds_settings["d"]) if self.needs_lds else [],
                 **{g: SCORERS[g](**self.scorer_specs[g]).names() for g in self.supervised}}
        return [c for g in self.groups for c in names[g]]

    def features(self, pv: pd.DataFrame, t: int, supervised: bool = True) -> pd.DataFrame:
        parts = []
        if {"context", "behavior"} & set(self.groups):
            parts.append(behavior_table(pv, t))
        if "ngram" in self.groups:
            parts.append(pd.DataFrame(ngram_features(list(pv["symbols"]), t),
                                      columns=ngram_names()))
        if "trigram" in self.groups:
            cols = self.tri_cols_[t]
            parts.append(pd.DataFrame(trigram_matrix(list(pv["symbols"]), t)[:, cols].toarray(),
                                      columns=[trigram_name(i) for i in cols]))
        if HMM_GROUPS & set(self.groups):
            seqs = hmm_symbols(pv, self.hmm_settings["alphabet"])
            filt = (self.hmm_.filter_macro(seqs) if hasattr(self.hmm_, "filter_macro")
                    else self.hmm_.filter(seqs))
            K = self._K()
            ll = self.hmm_.score_samples(seqs) / np.array([len(s) for s in seqs])
            parts.append(pd.DataFrame(np.column_stack([np.stack([f[-1] for f in filt]), ll]),
                                      columns=hmm_feature_names(K)))
            if "hmm_hist" in self.groups:
                parts.append(pd.DataFrame(hmm_history_features(filt, t),
                                          columns=hmm_history_names(K)))
        if self.needs_lds:
            means, ll = self.lds_.filter(encode_prefixes(pv))
            L = np.array([len(m) for m in means], float)
            parts.append(pd.DataFrame(np.column_stack([np.stack([m[-1] for m in means]),
                                                       np.stack([m.mean(0) for m in means]),
                                                       ll / L]),
                                      columns=lds_names(self.lds_settings["d"])))
        if supervised:
            parts += [self.scorers_[g].features(pv, t) for g in self.supervised]
            return pd.concat(parts, axis=1)[self.columns(t)]
        cols = [c for c in self.columns(t) if c not in self._supervised_cols()]
        return pd.concat(parts, axis=1)[cols] if parts else pd.DataFrame(index=pv.index)

    def _supervised_cols(self) -> list[str]:
        return [c for g in self.supervised for c in SCORERS[g](**self.scorer_specs[g]).names()]

    def _new_scorer(self, g: str):
        return SCORERS[g](**self.scorer_specs[g])

    def _crossfit(self, lr_runs, lr_y, prefixes) -> dict[int, pd.DataFrame]:
        issues = np.array(sorted(lr_runs["instance_id"].unique()))
        perm = np.random.default_rng(self.seed).permutation(len(issues))
        fold_of = dict(zip(issues[perm], np.arange(len(issues)) % self.crossfit_folds))
        fold = lr_runs["instance_id"].map(fold_of).to_numpy()
        cols = self._supervised_cols()
        oof = {t: pd.DataFrame(np.nan, index=range(len(lr_runs)), columns=cols) for t in prefixes}
        for k in range(self.crossfit_folds):
            tr, ho = fold != k, fold == k
            fitted = [self._new_scorer(g).fit(lr_runs[tr].reset_index(drop=True), lr_y[tr],
                                             prefixes, self.seed, getattr(self, "hmm_", None))
                      for g in self.supervised]
            for t in prefixes:
                rows = np.flatnonzero(ho & (lr_runs["n_steps"] > t).to_numpy())
                if not len(rows):
                    continue
                pv = prefix_view(lr_runs.iloc[rows], t)
                vals = pd.concat([f.features(pv, t) for f in fitted], axis=1)
                oof[t].iloc[rows] = vals[cols].to_numpy()
        return oof

    def fit(self, runs, y_fail, prefixes):
        y_fail = np.asarray(y_fail, bool)
        if self.needs_lds:
            self.lds_ = fit_or_load_lds(runs if self.hmm_data == "all" else
                                        runs[runs["split"] == "hmm_train"],
                                        self.lds_settings, self.seed)
            if not self.needs_hmm:
                use = (np.ones(len(runs), bool) if self.hmm_data == "all"
                       else (runs["split"] == "clf_train").to_numpy())
        if self.needs_hmm and self.hmm_data == "all":
            self.hmm_ = fit_or_load_hmm(runs, self.hmm_settings, self.seed)
            use = np.ones(len(runs), bool)
        elif self.needs_hmm:
            self.hmm_ = fit_or_load_hmm(runs[runs["split"] == "hmm_train"],
                                        self.hmm_settings, self.seed)
            use = (runs["split"] == "clf_train").to_numpy()
        elif not self.needs_lds:
            use = np.ones(len(runs), bool) if self.fit_on == "all" else \
                (runs["split"] == "clf_train").to_numpy()
        lr_runs, lr_y = runs[use].reset_index(drop=True), y_fail[use]
        if self.supervised:
            oof = self._crossfit(lr_runs, lr_y, prefixes)
            self.scorers_ = {g: self._new_scorer(g).fit(lr_runs, lr_y, prefixes, self.seed,
                                                       getattr(self, "hmm_", None))
                             for g in self.supervised}
        self.per_t_ = {}
        stack_X, stack_y = [], []
        for t in prefixes:
            alive = (lr_runs["n_steps"] > t).to_numpy()
            pv = prefix_view(lr_runs[alive], t)
            if "trigram" in self.groups:
                present = (trigram_matrix(list(pv["symbols"]), t) > 0).mean(axis=0).A1
                self.tri_cols_[t] = np.flatnonzero(present >= self.trigram_min_frac)
            if self.supervised:
                X = self.features(pv, t, supervised=False)
                X = pd.concat([X.reset_index(drop=True),
                               oof[t][alive].reset_index(drop=True)], axis=1)[self.columns(t)]
            else:
                X = self.features(pv, t)
            scaler = WithinModelScaler().fit(X, pv["model_name"]) if self.normalize else None
            Xn = scaler.transform(X, pv["model_name"]) if scaler else X.to_numpy(float)
            if self.landmark:
                stack_X.append(self._with_t(Xn, t))
                stack_y.append(lr_y[alive])
                self.per_t_[t] = (scaler, None)
            else:
                self.per_t_[t] = (scaler, self._new_clf().fit(Xn, lr_y[alive]))
        if self.landmark:
            self.landmark_clf_ = self._new_clf().fit(np.vstack(stack_X), np.concatenate(stack_y))
        return self

    def _new_clf(self):
        if self.classifier == "hgb":
            return HistGradientBoostingClassifier(**{**HGB_DEFAULTS, **self.clf_params},
                                                  random_state=self.seed)
        return LogisticRegression(C=self.C, max_iter=2000, class_weight=self.class_weight,
                                  **self.clf_params)

    def _with_t(self, Xn: np.ndarray, t: int) -> np.ndarray:
        lt = np.full((len(Xn), 1), np.log(t) - np.log(10.0))
        if self.classifier == "hgb":
            return np.hstack([Xn, lt])
        return np.hstack([Xn, Xn * lt, lt, lt ** 2])

    def predict_proba(self, prefix_runs, t):
        scaler, clf = self.per_t_[t]
        X = self.features(prefix_runs, t)
        Xn = scaler.transform(X, prefix_runs["model_name"]) if scaler else X.to_numpy(float)
        if self.landmark:
            return self.landmark_clf_.predict_proba(self._with_t(Xn, t))[:, 1]
        return clf.predict_proba(Xn)[:, 1]

    def coefficients(self) -> pd.DataFrame | None:
        if self.classifier != "lr" or self.landmark:
            return None
        return pd.concat([pd.DataFrame({"method": self.name, "t": t, "feature": self.columns(t),
                                        "coef": clf.coef_[0]})
                          for t, (_, clf) in self.per_t_.items()], ignore_index=True)

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "type": "feature_lr", "groups": self.groups, "C": self.C,
                "class_weight": self.class_weight, "normalize_within_model": self.normalize,
                "fit_on": (self.fit_on if not self.needs_hmm else
                           "clf_train (hmm: hmm_train)" if self.hmm_data == "split" else
                           "all (hmm: all)"),
                **({"hmm_data": self.hmm_data} if self.needs_hmm else {}),
                "classifier": self.classifier, "clf_params": self.clf_params,
                "landmark": self.landmark,
                **({"scorers": {g: self.scorers_[g].get_config() for g in self.supervised},
                    "crossfit_folds": self.crossfit_folds} if self.supervised else {}),
                "hmm": self.hmm_settings, "seed": self.seed,
                **({"trigram_min_frac": self.trigram_min_frac,
                    "n_trigrams_kept": {int(t): len(c) for t, c in self.tri_cols_.items()}}
                   if "trigram" in self.groups else {})}
