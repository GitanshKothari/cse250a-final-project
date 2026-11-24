from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..hmm import CategoricalHMM, _batches, _pad
from ..parsing import SYMBOLS

M = len(SYMBOLS)



def context_counts(seqs: list, order: int) -> sparse.csr_matrix:
    n_ctx = (M + 1) ** order
    rows, cols = [], []
    for n, s in enumerate(seqs):
        s = np.asarray(s, np.int64)
        if not len(s):
            continue
        padded = np.concatenate([np.full(order, M, np.int64), s])
        ctx = np.zeros(len(s), np.int64)
        for k in range(order):
            ctx = ctx * (M + 1) + padded[k: k + len(s)]
        rows.append(np.full(len(s), n))
        cols.append(ctx * M + s)
    if not rows:
        return sparse.csr_matrix((len(seqs), n_ctx * M))
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(seqs), n_ctx * M))


class MarkovScorer:
    groups = ("markov",)

    def __init__(self, order: int = 1, alpha: float = 1.0):
        self.order, self.alpha = order, alpha

    def names(self) -> list[str]:
        return [f"markov{self.order}_llr"]

    def _log_table(self, C: sparse.csr_matrix) -> np.ndarray:
        cnt = np.asarray(C.sum(0)).ravel().reshape(-1, M) + self.alpha
        return np.log(cnt / cnt.sum(1, keepdims=True)).ravel()

    def fit(self, runs: pd.DataFrame, y_fail: np.ndarray, prefixes, seed: int = 0,
            init_hmm=None) -> "MarkovScorer":
        self.w_ = {}
        for t in prefixes:
            alive = (runs["n_steps"] > t).to_numpy()
            C = context_counts([s[:t] for s in runs["symbols"][alive]], self.order)
            y = y_fail[alive]
            self.w_[t] = self._log_table(C[y]) - self._log_table(C[~y])
        return self

    def features(self, pv: pd.DataFrame, t: int) -> pd.DataFrame:
        C = context_counts(list(pv["symbols"]), self.order)
        return pd.DataFrame({self.names()[0]: C @ self.w_[t]})

    def get_config(self) -> dict:
        return {"type": "markov", "order": self.order, "alpha": self.alpha}



class ClassHMMScorer:
    groups = ("cchmm",)

    def __init__(self, n_iter: int = 30, tol: float = 1e-4, pseudocount: float = 0.01,
                 max_runs_per_class: int = 6000, max_len: int = 150):
        self.n_iter, self.tol, self.pc = n_iter, tol, pseudocount
        self.max_runs_per_class, self.max_len = max_runs_per_class, max_len

    def names(self) -> list[str]:
        return ["cchmm_llr", "cchmm_llr_per_step"]

    def _hmm(self, c: int) -> CategoricalHMM:
        h = CategoricalHMM(n_states=self.B_.shape[0], n_symbols=M)
        return h.set_params(self.pi_[c], self.A_[c], self.B_)

    def fit(self, runs, y_fail, prefixes, seed: int = 0, init_hmm=None) -> "ClassHMMScorer":
        if init_hmm is None:
            raise ValueError("cchmm needs the unsupervised HMM as initialisation")
        rng = np.random.default_rng(seed)
        data = []
        for c, is_fail in enumerate([True, False]):
            idx = np.flatnonzero(y_fail == is_fail)
            if self.max_runs_per_class and len(idx) > self.max_runs_per_class:
                idx = rng.choice(idx, self.max_runs_per_class, replace=False)
            data.append(_pad([np.asarray(runs["symbols"].iloc[i][: self.max_len]) for i in idx]))
        K = init_hmm.n_states
        self.pi_ = [init_hmm.startprob_.copy() for _ in range(2)]
        self.A_ = [init_hmm.transmat_.copy() for _ in range(2)]
        self.B_ = init_hmm.emissionprob_.copy()
        self.history_ = []
        n_obs = sum(m.sum() for _, m in data)
        for _ in range(self.n_iter):
            e_tot, ll = np.zeros((K, M)), 0.0
            new_pi, new_A = [], []
            for c, (X, mask) in enumerate(data):
                h = self._hmm(c)
                s_acc, t_acc = np.zeros(K), np.zeros((K, K))
                for sl in _batches(len(X), h.batch_size):
                    s, t_, e, l = h._e_step(X[sl], mask[sl])
                    s_acc += s
                    t_acc += t_
                    e_tot += e
                    ll += l
                new_pi.append((s_acc + self.pc) / (s_acc + self.pc).sum())
                new_A.append((t_acc + self.pc) / (t_acc + self.pc).sum(1, keepdims=True))
            self.pi_, self.A_ = new_pi, new_A
            self.B_ = (e_tot + self.pc) / (e_tot + self.pc).sum(1, keepdims=True)
            self.history_.append(ll / n_obs)
            if len(self.history_) > 1 and abs(self.history_[-1] - self.history_[-2]) < \
                    self.tol * abs(self.history_[-2]):
                break
        return self

    def features(self, pv: pd.DataFrame, t: int) -> pd.DataFrame:
        seqs = [np.asarray(s) for s in pv["symbols"]]
        llr = self._hmm(0).score_samples(seqs) - self._hmm(1).score_samples(seqs)
        L = np.array([len(s) for s in seqs], float)
        return pd.DataFrame({"cchmm_llr": llr, "cchmm_llr_per_step": llr / L})

    def transition_contrast(self) -> np.ndarray:
        return np.log(self.A_[0] / self.A_[1])

    def get_config(self) -> dict:
        return {"type": "cchmm", "n_iter": self.n_iter, "pseudocount": self.pc,
                "max_runs_per_class": self.max_runs_per_class, "max_len": self.max_len,
                "em_iters_used": len(getattr(self, "history_", []))}


SCORERS = {"markov": MarkovScorer, "cchmm": ClassHMMScorer}
