from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


def _pad(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([len(s) for s in sequences])
    if (lengths == 0).any():
        raise ValueError("Empty sequence passed to HMM.")
    N, T = len(sequences), int(lengths.max())
    X = np.zeros((N, T), dtype=np.int64)
    mask = np.zeros((N, T), dtype=bool)
    for n, s in enumerate(sequences):
        X[n, : len(s)] = s
        mask[n, : len(s)] = True
    return X, mask


def _batches(n: int, size: int):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


@dataclass
class CategoricalHMM:
    n_states: int
    n_symbols: int
    n_iter: int = 100
    tol: float = 1e-4
    n_init: int = 3
    pseudocount: float = 1e-2
    batch_size: int = 2000
    random_state: int | None = 0
    verbose: bool = False

    startprob_: np.ndarray | None = field(default=None, init=False)
    transmat_: np.ndarray | None = field(default=None, init=False)
    emissionprob_: np.ndarray | None = field(default=None, init=False)
    history_: list[float] = field(default_factory=list, init=False)
    restart_ll_: list[float] = field(default_factory=list, init=False)
    restart_iters_: list[int] = field(default_factory=list, init=False)

    def _random_params(self, rng: np.random.Generator):
        K, M = self.n_states, self.n_symbols
        pi = rng.dirichlet(np.ones(K))
        A = rng.dirichlet(np.ones(K), size=K) + 2.0 * np.eye(K)
        A /= A.sum(1, keepdims=True)
        B = rng.dirichlet(np.ones(M), size=K)
        return pi, A, B

    def set_params(self, pi: np.ndarray, A: np.ndarray, B: np.ndarray) -> "CategoricalHMM":
        self.startprob_, self.transmat_, self.emissionprob_ = (
            np.asarray(pi, float), np.asarray(A, float), np.asarray(B, float))
        return self

    def _forward(self, X: np.ndarray, mask: np.ndarray):
        pi, A, B = self.startprob_, self.transmat_, self.emissionprob_
        N, T = X.shape
        alpha = np.empty((N, T, self.n_states))
        c = np.ones((N, T))
        a = pi[None, :] * B[:, X[:, 0]].T
        c[:, 0] = a.sum(1)
        alpha[:, 0] = a / c[:, 0, None]
        for t in range(1, T):
            a = (alpha[:, t - 1] @ A) * B[:, X[:, t]].T
            s = a.sum(1)
            m = mask[:, t]
            c[m, t] = s[m]
            alpha[:, t] = np.where(m[:, None], a / np.where(m, s, 1.0)[:, None], alpha[:, t - 1])
        loglik = (np.log(c) * mask).sum(1)
        return alpha, c, loglik

    def _backward(self, X: np.ndarray, mask: np.ndarray, c: np.ndarray):
        A, B = self.transmat_, self.emissionprob_
        N, T = X.shape
        beta = np.ones((N, T, self.n_states))
        for t in range(T - 2, -1, -1):
            b = (B[:, X[:, t + 1]].T * beta[:, t + 1]) @ A.T / c[:, t + 1, None]
            beta[:, t] = np.where(mask[:, t + 1, None], b, 1.0)
        return beta

    def _e_step(self, X, mask):
        K, M = self.n_states, self.n_symbols
        A, B = self.transmat_, self.emissionprob_
        alpha, c, loglik = self._forward(X, mask)
        beta = self._backward(X, mask, c)
        gamma = alpha * beta * mask[..., None]

        start = gamma[:, 0].sum(0)
        trans = np.zeros((K, K))
        for t in range(X.shape[1] - 1):
            m = mask[:, t + 1, None]
            right = B[:, X[:, t + 1]].T * beta[:, t + 1] / c[:, t + 1, None] * m
            trans += (alpha[:, t].T @ right)
        trans *= A
        emit = np.zeros((K, M))
        for v in range(M):
            emit[:, v] = gamma[X == v].sum(0)
        return start, trans, emit, loglik.sum()

    def fit(self, sequences: Sequence[np.ndarray]) -> "CategoricalHMM":
        X_all, mask_all = _pad(sequences)
        n_obs = mask_all.sum()
        rng = np.random.default_rng(self.random_state)
        best = (-np.inf, None, None)
        self.restart_ll_, self.restart_iters_ = [], []
        init = getattr(self, "init_params", None)
        for restart in range(self.n_init):
            if restart == 0 and init is not None:
                self.set_params(*init)
            else:
                self.set_params(*self._random_params(rng))
            history: list[float] = []
            for it in range(self.n_iter):
                K, M = self.n_states, self.n_symbols
                s_acc, t_acc, e_acc, ll = np.zeros(K), np.zeros((K, K)), np.zeros((K, M)), 0.0
                for sl in _batches(len(X_all), self.batch_size):
                    s, t_, e, l = self._e_step(X_all[sl], mask_all[sl])
                    s_acc += s
                    t_acc += t_
                    e_acc += e
                    ll += l
                history.append(ll / n_obs)
                self._m_step(s_acc, t_acc, e_acc)
                if self.verbose:
                    print(f"  restart {restart} iter {it:3d}  mean loglik/step {history[-1]:.5f}")
                if it > 0 and abs(history[-1] - history[-2]) < self.tol * abs(history[-2]):
                    break
            final = self.score(sequences) / n_obs
            self.restart_ll_.append(float(final))
            self.restart_iters_.append(len(history))
            if final > best[0]:
                best = (final, (self.startprob_, self.transmat_, self.emissionprob_), history)
        self.set_params(*best[1])
        self.history_ = best[2]
        return self

    def _m_step(self, s_acc: np.ndarray, t_acc: np.ndarray, e_acc: np.ndarray) -> None:
        pc = self.pseudocount
        self.startprob_ = (s_acc + pc) / (s_acc + pc).sum()
        self.transmat_ = (t_acc + pc) / (t_acc + pc).sum(1, keepdims=True)
        self.emissionprob_ = (e_acc + pc) / (e_acc + pc).sum(1, keepdims=True)

    def _check(self):
        if self.startprob_ is None:
            raise RuntimeError("Model is not fitted.")

    def score_samples(self, sequences: Sequence[np.ndarray]) -> np.ndarray:
        self._check()
        X, mask = _pad(sequences)
        out = np.empty(len(sequences))
        for sl in _batches(len(X), self.batch_size):
            out[sl] = self._forward(X[sl], mask[sl])[2]
        return out

    def score(self, sequences: Sequence[np.ndarray]) -> float:
        return float(self.score_samples(sequences).sum())

    def filter(self, sequences: Sequence[np.ndarray]) -> list[np.ndarray]:
        self._check()
        X, mask = _pad(sequences)
        out: list[np.ndarray] = []
        for sl in _batches(len(X), self.batch_size):
            alpha, _, _ = self._forward(X[sl], mask[sl])
            out.extend(alpha[i, : mask[sl][i].sum()] for i in range(alpha.shape[0]))
        return out

    def smooth(self, sequences: Sequence[np.ndarray]) -> list[np.ndarray]:
        self._check()
        X, mask = _pad(sequences)
        out: list[np.ndarray] = []
        for sl in _batches(len(X), self.batch_size):
            alpha, c, _ = self._forward(X[sl], mask[sl])
            gamma = alpha * self._backward(X[sl], mask[sl], c)
            out.extend(gamma[i, : mask[sl][i].sum()] for i in range(gamma.shape[0]))
        return out

    def viterbi(self, sequences: Sequence[np.ndarray]) -> list[np.ndarray]:
        self._check()
        logpi = np.log(self.startprob_)
        logA, logB = np.log(self.transmat_), np.log(self.emissionprob_)
        paths: list[np.ndarray] = []
        X_all, mask_all = _pad(sequences)
        for sl in _batches(len(X_all), self.batch_size):
            X, mask = X_all[sl], mask_all[sl]
            N, T = X.shape
            delta = logpi[None] + logB[:, X[:, 0]].T
            psi = np.zeros((N, T, self.n_states), dtype=np.int64)
            for t in range(1, T):
                cand = delta[:, :, None] + logA[None]
                psi[:, t] = cand.argmax(1)
                new = cand.max(1) + logB[:, X[:, t]].T
                delta = np.where(mask[:, t, None], new, delta)
            lengths = mask.sum(1)
            for n in range(N):
                L = lengths[n]
                path = np.empty(L, dtype=np.int64)
                path[-1] = delta[n].argmax()
                for t in range(L - 1, 0, -1):
                    path[t - 1] = psi[n, t, path[t]]
                paths.append(path)
        return paths

    def n_free_params(self) -> int:
        K, M = self.n_states, self.n_symbols
        return (K - 1) + K * (K - 1) + K * (M - 1)

    def bic(self, sequences: Sequence[np.ndarray]) -> float:
        n_obs = sum(len(s) for s in sequences)
        return -2.0 * self.score(sequences) + self.n_free_params() * np.log(n_obs)

    def save(self, path: str | Path) -> None:
        self._check()
        np.savez(path, startprob=self.startprob_, transmat=self.transmat_,
                 emissionprob=self.emissionprob_, history=np.array(self.history_),
                 restart_ll=np.array(self.restart_ll_), restart_iters=np.array(self.restart_iters_))

    @classmethod
    def load(cls, path: str | Path) -> "CategoricalHMM":
        d = np.load(path)
        K, M = d["emissionprob"].shape
        m = cls(n_states=K, n_symbols=M)
        m.set_params(d["startprob"], d["transmat"], d["emissionprob"])
        m.history_ = d["history"].tolist()
        if "restart_ll" in d:
            m.restart_ll_ = d["restart_ll"].tolist()
            m.restart_iters_ = d["restart_iters"].tolist()
        return m



class PhaseHMM(CategoricalHMM):
    def __init__(self, n_macro: int, n_phases: int, n_symbols: int, **kw):
        super().__init__(n_states=n_macro * n_phases, n_symbols=n_symbols, **kw)
        self.n_macro, self.n_phases = n_macro, n_phases
        K, P = n_macro, n_phases
        idx = lambda k, i: k * P + i
        mask = np.zeros((K * P, K * P), bool)
        for k in range(K):
            for i in range(P):
                mask[idx(k, i), idx(k, i)] = True
                if i + 1 < P:
                    mask[idx(k, i), idx(k, i + 1)] = True
                for j in range(K):
                    if j != k:
                        mask[idx(k, i), idx(j, 0)] = True
        self.trans_mask_ = mask
        self.start_mask_ = np.array([i == 0 for k in range(K) for i in range(P)])
        self.macro_of_ = np.repeat(np.arange(K), P)

    def _random_params(self, rng):
        K, P, M = self.n_macro, self.n_phases, self.n_symbols
        pi = np.where(self.start_mask_, rng.dirichlet(np.ones(K * P)), 0.0)
        A = (rng.dirichlet(np.ones(K * P), size=K * P) + 2.0 * np.eye(K * P)) * self.trans_mask_
        B = np.repeat(rng.dirichlet(np.ones(M), size=K), P, axis=0)
        return pi / pi.sum(), A / A.sum(1, keepdims=True), B

    def _m_step(self, s_acc, t_acc, e_acc):
        pc = self.pseudocount
        s = np.where(self.start_mask_, s_acc + pc, 0.0)
        t = np.where(self.trans_mask_, t_acc + pc, 0.0)
        e = np.zeros((self.n_macro, self.n_symbols))
        np.add.at(e, self.macro_of_, e_acc)
        e = (e + pc) / (e + pc).sum(1, keepdims=True)
        self.startprob_ = s / s.sum()
        self.transmat_ = t / t.sum(1, keepdims=True)
        self.emissionprob_ = e[self.macro_of_]

    @classmethod
    def params_from_hmm(cls, base: CategoricalHMM, n_phases: int):
        K, P = base.n_states, n_phases
        a = base.transmat_
        pi = np.zeros(K * P)
        A = np.zeros((K * P, K * P))
        for k in range(K):
            pi[k * P] = base.startprob_[k]
            for i in range(P):
                r = k * P + i
                A[r, r] = a[k, k]
                share = 0.5 if i + 1 < P else 1.0
                if i + 1 < P:
                    A[r, r + 1] = (1 - a[k, k]) * 0.5
                for j in range(K):
                    if j != k:
                        A[r, j * P] = a[k, j] * share
        return pi, A / A.sum(1, keepdims=True), np.repeat(base.emissionprob_, P, axis=0)

    def filter_macro(self, sequences):
        G = np.eye(self.n_macro)[self.macro_of_]
        return [f @ G for f in self.filter(sequences)]

    def dwell_pmf(self, max_d: int = 60) -> np.ndarray:
        K, P, A = self.n_macro, self.n_phases, self.transmat_
        out = np.zeros((K, max_d))
        for k in range(K):
            sl = slice(k * P, (k + 1) * P)
            Q = A[sl, sl]
            exit_ = 1.0 - Q.sum(1)
            v = np.eye(P)[0]
            for d in range(max_d):
                out[k, d] = v @ exit_
                v = v @ Q
        return out

    def n_free_params(self) -> int:
        return int(self.start_mask_.sum() - 1 + (self.trans_mask_.sum() - self.n_states)
                   + self.n_macro * (self.n_symbols - 1))

    def save(self, path):
        self._check()
        np.savez(path, startprob=self.startprob_, transmat=self.transmat_,
                 emissionprob=self.emissionprob_, history=np.array(self.history_),
                 restart_ll=np.array(self.restart_ll_), restart_iters=np.array(self.restart_iters_),
                 n_macro=self.n_macro, n_phases=self.n_phases)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(int(d["n_macro"]), int(d["n_phases"]), d["emissionprob"].shape[1])
        m.set_params(d["startprob"], d["transmat"], d["emissionprob"])
        m.history_ = d["history"].tolist()
        m.restart_ll_, m.restart_iters_ = d["restart_ll"].tolist(), d["restart_iters"].tolist()
        return m
