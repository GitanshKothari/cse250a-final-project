from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ["test", "val", "hmm_train", "clf_train"]
SOLVE_RATE_BINS = [-0.01, 0.0, 0.25, 0.5, 0.75, 1.0]


def split_fractions(test: float, val: float, hmm_frac_of_train: float) -> dict[str, float]:
    train = 1.0 - test - val
    return {"test": test, "val": val, "hmm_train": train * hmm_frac_of_train,
            "clf_train": train * (1.0 - hmm_frac_of_train)}


def _assign_issues(rate: pd.Series, test: float, val: float, hmm_frac_of_train: float,
                   seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    rate = rate.sort_index()
    strata = pd.cut(rate, bins=SOLVE_RATE_BINS, labels=False)
    assignment: dict[str, str] = {}
    for _, ids in rate.groupby(strata).groups.items():
        ids = np.array(sorted(ids))
        rng.shuffle(ids)
        n = len(ids)
        n_test, n_val = int(round(n * test)), int(round(n * val))
        n_hmm = int(round((n - n_test - n_val) * hmm_frac_of_train))
        cuts = np.cumsum([n_test, n_val, n_hmm])
        for name, part in zip(SPLIT_NAMES, np.split(ids, cuts)):
            assignment.update({i: name for i in part})
    return assignment


def make_splits(df: pd.DataFrame, test: float, val: float, hmm_frac_of_train: float,
                seed: int) -> pd.Series:
    rate = df.groupby("instance_id")["target"].mean()
    assignment = _assign_issues(rate, test, val, hmm_frac_of_train, seed)
    return df["instance_id"].map(assignment).rename("split")


def build_split_table(meta: pd.DataFrame, test: float, val: float, hmm_frac_of_train: float,
                      seed: int) -> pd.DataFrame:
    g = meta.groupby("instance_id")["target"]
    tab = pd.DataFrame({"n_runs": g.size(), "solve_rate": g.mean().round(4)})
    assignment = _assign_issues(g.mean(), test, val, hmm_frac_of_train, seed)
    tab["split"] = tab.index.map(assignment)
    tab["repo"] = [i.rsplit("-", 1)[0] for i in tab.index]
    return tab.reset_index()[["instance_id", "split", "repo", "n_runs", "solve_rate"]]


def hash_split(instance_id: str, fractions: dict[str, float]) -> str:
    u = int(hashlib.sha256(instance_id.encode()).hexdigest()[:12], 16) / 16**12
    edge = 0.0
    for name in SPLIT_NAMES:
        edge += fractions[name]
        if u < edge:
            return name
    return SPLIT_NAMES[-1]


def load_split_table(path: Path) -> pd.Series:
    tab = pd.read_csv(path)
    if tab["instance_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate instance_id")
    if not set(tab["split"]) <= set(SPLIT_NAMES):
        raise ValueError(f"{path}: unknown split names {set(tab['split']) - set(SPLIT_NAMES)}")
    return tab.set_index("instance_id")["split"]


def assign_splits(instance_ids: pd.Series, frozen: pd.Series,
                  fractions: dict[str, float]) -> tuple[pd.Series, int]:
    split = instance_ids.map(frozen)
    missing = instance_ids[split.isna()].unique()
    if len(missing):
        split = split.fillna(instance_ids.map({i: hash_split(i, fractions) for i in missing}))
    return split.rename("split"), len(missing)



TRAIN_SPLITS = ["hmm_train", "clf_train"]


def build_folds(split_table: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    tr = split_table[split_table["split"].isin(TRAIN_SPLITS)].sort_values("instance_id")
    rng = np.random.default_rng(seed)
    strata = pd.cut(tr["solve_rate"], bins=SOLVE_RATE_BINS, labels=False)
    rows = []
    offset = 0
    for _, ids in tr.groupby(strata)["instance_id"]:
        ids = ids.to_numpy().copy()
        rng.shuffle(ids)
        rows += [(i, (offset + k) % n_folds) for k, i in enumerate(ids)]
        offset += len(ids)
    return pd.DataFrame(rows, columns=["instance_id", "fold"]).sort_values("instance_id")


def load_folds(path: Path) -> pd.Series:
    f = pd.read_csv(path)
    if f["instance_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate instance_id")
    return f.set_index("instance_id")["fold"]
