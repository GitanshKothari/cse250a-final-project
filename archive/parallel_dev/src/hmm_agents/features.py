from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm import CategoricalHMM
from .markov import MarkovChain
from .parsing import ACTIONS, SYMBOLS

STICKY_THRESHOLD = 0.9

BEHAVIOR_FEATURES = [
    "err_count", "err_rate", "err_streak", "repeat_cmds", "n_viewed_files",
    "n_edited_files", "any_test_passed", "first_edit_step", "has_edited",
    *[f"frac_{a}" for a in ACTIONS],
]
CONTEXT_FEATURES = ["log_issue_len"]


def _streak(errors: list[bool]) -> int:
    n = 0
    for e in reversed(errors):
        if not e:
            break
        n += 1
    return n


def behavior_features(row: pd.Series, t: int) -> dict[str, float]:
    acts, errs, cmds = row["actions"][:t], list(row["errors"][:t]), row["commands"][:t]
    edits = [i for i, a in enumerate(acts) if a == "edit"]
    return {
        "err_count": sum(errs),
        "err_rate": sum(errs) / t,
        "err_streak": _streak(errs),
        "repeat_cmds": len(cmds) - len(set(cmds)),
        "n_viewed_files": len({f for f in row["viewed_files"][:t] if f}),
        "n_edited_files": len({f for f in row["edited_files"][:t] if f}),
        "any_test_passed": float(any(row["test_passed"][:t])),
        "first_edit_step": (edits[0] + 1) if edits else t + 1,
        "has_edited": float(bool(edits)),
        **{f"frac_{a}": sum(x == a for x in acts) / t for a in ACTIONS},
        "log_issue_len": np.log1p(row["issue_len"]),
    }


def hmm_features(hmm: CategoricalHMM, sequences: list[np.ndarray], t: int) -> np.ndarray:
    prefixes = [np.asarray(s[:t]) for s in sequences]
    filt = np.stack([f[-1] for f in hmm.filter(prefixes)])
    ll = hmm.score_samples(prefixes) / np.array([len(p) for p in prefixes])
    return np.column_stack([filt, ll])


def hmm_feature_names(K: int) -> list[str]:
    return [f"p_state_{k}" for k in range(K)] + ["prefix_loglik_per_step"]


def sticky_states(hmm: CategoricalHMM, threshold: float = STICKY_THRESHOLD) -> np.ndarray:
    return np.flatnonzero(np.diag(hmm.transmat_) >= threshold)


def hmm_memory_feature_names(K: int) -> list[str]:
    return [f"mean_p_state_{k}" for k in range(K)] + ["max_p_sticky"]


def hmm_memory_features(hmm: CategoricalHMM, sequences: list[np.ndarray], t: int) -> np.ndarray:
    filt = hmm.filter([np.asarray(s[:t]) for s in sequences])
    sticky = sticky_states(hmm)
    mean = np.stack([f.mean(0) for f in filt])
    max_sticky = np.array([f[:, sticky].sum(1).max() if len(sticky) else 0.0 for f in filt])
    return np.column_stack([mean, max_sticky])


MARKOV_FEATURES = ([f"last_{s}" for s in SYMBOLS] + [f"share_{s}" for s in SYMBOLS]
                   + ["mc_loglik_per_step", "repeat_rate"])


def markov_features(mc: MarkovChain, sequences: list[np.ndarray], t: int) -> np.ndarray:
    M = len(SYMBOLS)
    prefixes = [np.asarray(s[:t]) for s in sequences]
    last = np.eye(M)[[p[-1] for p in prefixes]]
    share = np.stack([np.bincount(p, minlength=M) / len(p) for p in prefixes])
    ll = mc.score_samples(prefixes) / np.array([len(p) for p in prefixes])
    rep = np.array([(p[1:] == p[:-1]).mean() if len(p) > 1 else 0.0 for p in prefixes])
    return np.column_stack([last, share, ll, rep])


def build_prefix_table(df: pd.DataFrame, hmm: CategoricalHMM, t: int,
                       mc: MarkovChain | None = None) -> pd.DataFrame:
    alive = df[df["n_steps"] > t].reset_index(drop=True)
    seqs = list(alive["symbols"])
    beh = pd.DataFrame([behavior_features(r, t) for _, r in alive.iterrows()])
    parts = [alive[["run_id", "instance_id", "model_name", "target", "split"]], beh,
             pd.DataFrame(hmm_features(hmm, seqs, t), columns=hmm_feature_names(hmm.n_states)),
             pd.DataFrame(hmm_memory_features(hmm, seqs, t),
                          columns=hmm_memory_feature_names(hmm.n_states))]
    if mc is not None:
        parts.append(pd.DataFrame(markov_features(mc, seqs, t), columns=MARKOV_FEATURES))
    return pd.concat(parts, axis=1)


def normalize_within_model(train: pd.DataFrame, others: list[pd.DataFrame],
                           cols: list[str]) -> list[pd.DataFrame]:
    g_mu, g_sd = train[cols].mean(), train[cols].std().replace(0, 1).fillna(1)
    stats = train.groupby("model_name")[cols].agg(["mean", "std"])
    out = []
    for frame in [train, *others]:
        frame = frame.copy()
        frame[cols] = frame[cols].astype(float)
        for model, idx in frame.groupby("model_name").groups.items():
            if model in stats.index:
                mu = stats.loc[model].xs("mean", level=1)
                sd = stats.loc[model].xs("std", level=1).replace(0, 1).fillna(1)
            else:
                mu, sd = g_mu, g_sd
            frame.loc[idx, cols] = ((frame.loc[idx, cols] - mu) / sd).to_numpy()
        out.append(frame)
    return out
