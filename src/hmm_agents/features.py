from __future__ import annotations

import numpy as np
import pandas as pd

from .parsing import ACTIONS

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
        "any_test_passed": float(any(row["test_passed"][:t]) or any(row["success_msg"][:t])),
        "first_edit_step": (edits[0] + 1) if edits else t + 1,
        "has_edited": float(bool(edits)),
        **{f"frac_{a}": sum(x == a for x in acts) / t for a in ACTIONS},
        "log_issue_len": np.log1p(row["issue_len"]),
    }


_BEH_CACHE: dict[tuple[int, int], dict[str, float]] = {}


def clear_feature_cache() -> None:
    _BEH_CACHE.clear()


def behavior_table(runs: pd.DataFrame, t: int) -> pd.DataFrame:
    out = []
    for r in runs.itertuples(index=False):
        key = (int(r.run_id), t)
        if key not in _BEH_CACHE:
            _BEH_CACHE[key] = behavior_features(r._asdict(), t)
        out.append(_BEH_CACHE[key])
    return pd.DataFrame(out, columns=BEHAVIOR_FEATURES + CONTEXT_FEATURES)


def ngram_names() -> list[str]:
    from .parsing import SYMBOLS
    S = len(SYMBOLS)
    return ([f"uni_{SYMBOLS[i]}" for i in range(S)]
            + [f"bi_{SYMBOLS[i]}>{SYMBOLS[j]}" for i in range(S) for j in range(S)])


def ngram_features(symbol_prefixes: list, t: int) -> np.ndarray:
    from .parsing import SYMBOLS
    S = len(SYMBOLS)
    out = np.zeros((len(symbol_prefixes), S + S * S))
    for r, s in enumerate(symbol_prefixes):
        s = np.asarray(s[:t], int)
        out[r, :S] = np.bincount(s, minlength=S) / len(s)
        if len(s) > 1:
            out[r, S:] = np.bincount(s[:-1] * S + s[1:], minlength=S * S) / (len(s) - 1)
    return out


def trigram_matrix(symbol_prefixes: list, t: int):
    from scipy import sparse

    from .parsing import SYMBOLS
    S = len(SYMBOLS)
    rows, cols, vals = [], [], []
    for r, s in enumerate(symbol_prefixes):
        s = np.asarray(s[:t], int)
        if len(s) < 3:
            continue
        ids, cnt = np.unique(s[:-2] * S * S + s[1:-1] * S + s[2:], return_counts=True)
        rows += [r] * len(ids)
        cols += list(ids)
        vals += list(cnt / (len(s) - 2))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(symbol_prefixes), S ** 3))


def trigram_name(i: int) -> str:
    from .parsing import SYMBOLS
    S = len(SYMBOLS)
    return f"tri_{SYMBOLS[i // (S * S)]}>{SYMBOLS[(i // S) % S]}>{SYMBOLS[i % S]}"


def hmm_feature_names(K: int) -> list[str]:
    return [f"p_state_{k}" for k in range(K)] + ["prefix_loglik_per_step"]


ENTER_RUN = 3


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def history_from_filtered(filt: np.ndarray) -> np.ndarray:
    t, K = filt.shape
    am = filt.argmax(1)
    dwell = 1
    while dwell < t and am[t - 1 - dwell] == am[-1]:
        dwell += 1
    entered = [float(_longest_run(am == k) >= ENTER_RUN) for k in range(K)]
    return np.concatenate([filt.mean(0), filt.max(0), entered,
                           [dwell, (am[1:] != am[:-1]).sum() / t]])


def hmm_history_features(filtered: list[np.ndarray], t: int) -> np.ndarray:
    return np.stack([history_from_filtered(f[:t]) for f in filtered])


def hmm_history_names(K: int) -> list[str]:
    return ([f"occ_state_{k}" for k in range(K)] + [f"max_state_{k}" for k in range(K)]
            + [f"entered_state_{k}" for k in range(K)] + ["cur_dwell", "switch_rate"])
