from __future__ import annotations

import numpy as np
import pandas as pd

from .parsing import SYMBOLS

ENCODER_VERSION = 1
STEP_FEATURE_NAMES = ([f"sym_{s}" for s in SYMBOLS]
                      + ["no_match", "success_msg", "test_passed", "repeat_cmd",
                         "new_view", "new_edit", "log_pos"])
D = len(STEP_FEATURE_NAMES)


def encode_run(row, t: int | None = None) -> np.ndarray:
    syms = np.asarray(row["symbols"], int)
    T = len(syms) if t is None else min(t, len(syms))
    x = np.zeros((T, D), np.float32)
    x[np.arange(T), syms[:T]] = 1.0
    S = len(SYMBOLS)
    x[:, S] = np.asarray(row["no_match"][:T], np.float32)
    x[:, S + 1] = np.asarray(row["success_msg"][:T], np.float32)
    x[:, S + 2] = np.asarray(row["test_passed"][:T], np.float32)
    seen_cmd, seen_view, seen_edit = set(), set(), set()
    cmds, views, edits = row["commands"], row["viewed_files"], row["edited_files"]
    for i in range(T):
        c, v, e = cmds[i], views[i], edits[i]
        x[i, S + 3] = c in seen_cmd
        x[i, S + 4] = bool(v) and v not in seen_view
        x[i, S + 5] = bool(e) and e not in seen_edit
        seen_cmd.add(c)
        if v:
            seen_view.add(v)
        if e:
            seen_edit.add(e)
    x[:, S + 6] = np.log1p(np.arange(1, T + 1))
    return x


def encode_prefixes(prefix_runs: pd.DataFrame) -> list[np.ndarray]:
    return [encode_run(r) for r in prefix_runs.to_dict("records")]


def pad_batch(seqs: list[np.ndarray], max_len: int | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    if max_len is not None:
        seqs = [s[-max_len:] for s in seqs]
    T = max((len(s) for s in seqs), default=0)
    X = np.zeros((len(seqs), T, D), np.float32)
    mask = np.zeros((len(seqs), T), bool)
    for i, s in enumerate(seqs):
        X[i, : len(s)] = s
        mask[i, : len(s)] = True
    return X, mask



RICH_FLAGS = ["repeat_cmd", "new_file", "passed"]
N_RICH = len(SYMBOLS) * 2 ** len(RICH_FLAGS)


def rich_symbols(row, t: int | None = None) -> np.ndarray:
    x = encode_run(row, t)
    S = len(SYMBOLS)
    sym = x[:, :S].argmax(1)
    repeat = x[:, S + 3] > 0
    new_file = (x[:, S + 4] > 0) | (x[:, S + 5] > 0)
    passed = (x[:, S + 2] > 0) | (x[:, S + 1] > 0)
    return (sym * 8 + repeat * 4 + new_file * 2 + passed).astype(np.int64)
