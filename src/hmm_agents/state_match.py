from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

RARE = 0.01
DUP_EMIT = 0.15
DUP_STAY = 0.10
STICKY = 0.90


def hellinger(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    P, Q = np.atleast_2d(P), np.atleast_2d(Q)
    if P.shape[1] != Q.shape[1]:
        raise ValueError(f"different symbol counts: {P.shape[1]} vs {Q.shape[1]}")
    return np.sqrt(0.5 * ((np.sqrt(P)[:, None] - np.sqrt(Q)[None]) ** 2).sum(-1))


def common_symbols(B: np.ndarray, n_symbols: int) -> tuple[np.ndarray, np.ndarray]:
    kept = B[:, :n_symbols]
    mass = kept.sum(1)
    if (mass <= 0).any():
        raise ValueError("a state emits only dropped symbols; it has no common-symbol row")
    return kept / mass[:, None], 1.0 - mass


def match_states(trans_ref: np.ndarray, B_ref: np.ndarray, trans: np.ndarray,
                 B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = min(B_ref.shape[1], B.shape[1])
    if B_ref.shape[1] != n:
        B_ref = common_symbols(B_ref, n)[0]
    if B.shape[1] != n:
        B = common_symbols(B, n)[0]
    H = hellinger(B_ref, B)
    S = np.abs(np.diag(trans_ref)[:, None] - np.diag(trans)[None])
    r, c = linear_sum_assignment(H + S)
    return r, c, H[r, c], S[r, c]


def state_shares(hmm, sequences: Sequence[np.ndarray]) -> np.ndarray:
    paths = hmm.viterbi([np.asarray(s) for s in sequences])
    return np.bincount(np.concatenate(paths), minlength=hmm.n_states) / sum(map(len, paths))


def run_occupancy(hmm, sequences: Sequence[np.ndarray]) -> np.ndarray:
    paths = hmm.viterbi([np.asarray(s) for s in sequences])
    return np.stack([np.bincount(p, minlength=hmm.n_states) / len(p) for p in paths])


def distinctness(hmm, sequences: Sequence[np.ndarray], rare: float = RARE,
                 dup_emit: float = DUP_EMIT, dup_stay: float = DUP_STAY,
                 sticky: float = STICKY) -> dict:
    share = state_shares(hmm, sequences)
    A, B = hmm.transmat_, hmm.emissionprob_
    stay = np.diag(A)
    H = hellinger(B, B)
    K = len(share)
    pairs = [(H[i, j], abs(stay[i] - stay[j]), i, j) for i in range(K) for j in range(i + 1, K)]
    dups = [p for p in pairs if p[0] < dup_emit and p[1] < dup_stay]
    h, d, i, j = min(pairs) if pairs else (np.nan, np.nan, -1, -1)
    return {"K": K, "n_rare_modes": int((share < rare).sum()),
            "smallest_mode_share": float(share.min()),
            "n_near_duplicate_pairs": len(dups), "closest_pair": f"S{i}-S{j}",
            "closest_hellinger": float(h), "closest_stay_diff": float(d),
            "n_sticky": int((stay >= sticky).sum())}
