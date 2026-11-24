from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class MarkovChain:
    def __init__(self, n_symbols: int, pseudocount: float = 0.01):
        self.n_symbols = n_symbols
        self.pseudocount = pseudocount
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None

    def fit(self, sequences: Sequence[np.ndarray]) -> "MarkovChain":
        M, a = self.n_symbols, self.pseudocount
        start, trans = np.full(M, a), np.full((M, M), a)
        for s in sequences:
            s = np.asarray(s)
            if len(s) == 0:
                continue
            start[s[0]] += 1
            np.add.at(trans, (s[:-1], s[1:]), 1)
        self.startprob_ = start / start.sum()
        self.transmat_ = trans / trans.sum(1, keepdims=True)
        return self

    def score_samples(self, sequences: Sequence[np.ndarray]) -> np.ndarray:
        if self.startprob_ is None:
            raise RuntimeError("Model is not fitted.")
        with np.errstate(divide="ignore"):
            logpi, logA = np.log(self.startprob_), np.log(self.transmat_)
        out = np.zeros(len(sequences))
        for i, s in enumerate(sequences):
            s = np.asarray(s)
            if len(s):
                out[i] = logpi[s[0]] + logA[s[:-1], s[1:]].sum()
        return out

    def score(self, sequences: Sequence[np.ndarray]) -> float:
        return float(self.score_samples(sequences).sum())

    def save(self, path: str | Path) -> None:
        np.savez(path, startprob=self.startprob_, transmat=self.transmat_,
                 pseudocount=self.pseudocount)

    @classmethod
    def load(cls, path: str | Path) -> "MarkovChain":
        d = np.load(path)
        m = cls(n_symbols=len(d["startprob"]), pseudocount=float(d["pseudocount"]))
        m.startprob_, m.transmat_ = d["startprob"], d["transmat"]
        return m
