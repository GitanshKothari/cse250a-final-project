from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STEP_COLUMNS = ["symbols", "actions", "errors", "commands", "viewed_files", "edited_files",
                "test_passed", "no_match", "success_msg"]
VISIBLE_RUN_COLUMNS = ["run_id", "instance_id", "model_name", "issue_len"]


def prefix_view(runs: pd.DataFrame, t: int) -> pd.DataFrame:
    keep = [c for c in VISIBLE_RUN_COLUMNS + STEP_COLUMNS if c in runs.columns]
    out = runs[keep].reset_index(drop=True).copy()
    for c in STEP_COLUMNS:
        if c in out.columns:
            out[c] = [v[:t] for v in out[c]]
    out["t"] = t
    return out


def group_holdout(runs: pd.DataFrame, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    issues = np.array(sorted(runs["instance_id"].unique()))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(issues, size=max(1, int(round(frac * len(issues)))), replace=False))
    hold = runs["instance_id"].isin(held).to_numpy()
    return ~hold, hold


class FailurePredictor(ABC):
    name: str = "unnamed"
    stochastic: bool = True

    @abstractmethod
    def fit(self, runs: pd.DataFrame, y_fail: np.ndarray,
            prefixes: list[int]) -> "FailurePredictor":
        pass

    @abstractmethod
    def predict_proba(self, prefix_runs: pd.DataFrame, t: int) -> np.ndarray:
        pass

    def explain(self, prefix_runs: pd.DataFrame, t: int) -> list[dict] | None:
        return None

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "FailurePredictor":
        with open(path, "rb") as f:
            return pickle.load(f)
