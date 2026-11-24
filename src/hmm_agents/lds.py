from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LOG2PI = np.log(2 * np.pi)


def _pad(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    L = np.array([len(s) for s in seqs])
    if (L == 0).any():
        raise ValueError("empty sequence")
    X = np.zeros((len(seqs), L.max(), seqs[0].shape[1]))
    for i, s in enumerate(seqs):
        X[i, : len(s)] = s
    return X, L


@dataclass
class GaussianLDS:
    d: int = 4
    n_iter: int = 30
    tol: float = 1e-4
    r_floor: float = 1e-3
    random_state: int | None = 0

    mu0: np.ndarray | None = field(default=None, init=False)
    V0: np.ndarray | None = field(default=None, init=False)
    A: np.ndarray | None = field(default=None, init=False)
    Q: np.ndarray | None = field(default=None, init=False)
    C: np.ndarray | None = field(default=None, init=False)
    b: np.ndarray | None = field(default=None, init=False)
    R: np.ndarray | None = field(default=None, init=False)
    history_: list[float] = field(default_factory=list, init=False)

    def _init(self, X: np.ndarray, L: np.ndarray, rng) -> None:
        D, d = X.shape[2], self.d
        mask = np.arange(X.shape[1])[None] < L[:, None]
        Y = X[mask]
        self.b = Y.mean(0)
        U, s, Vt = np.linalg.svd(Y - self.b, full_matrices=False)
        rot = np.linalg.qr(rng.normal(size=(d, d)))[0]
        self.C = (Vt[:d].T * (s[:d] / np.sqrt(len(Y)))) @ rot
        self.R = np.maximum(Y.var(0) - (self.C ** 2).sum(1), self.r_floor)
        self.A, self.Q = 0.9 * np.eye(d), 0.1 * np.eye(d)
        self.mu0, self.V0 = np.zeros(d), np.eye(d)
        assert self.C.shape == (D, d)

    def _cov_pass(self, T: int):
        d = self.d
        Pp, Pf, K, Si, logdet = (np.empty((T, d, d)), np.empty((T, d, d)),
                                 np.empty((T, d, self.C.shape[0])),
                                 np.empty((T, self.C.shape[0], self.C.shape[0])), np.empty(T))
        P = self.V0
        for t in range(T):
            if t > 0:
                P = self.A @ Pf[t - 1] @ self.A.T + self.Q
            Pp[t] = P
            S = self.C @ P @ self.C.T + np.diag(self.R)
            Sinv = np.linalg.inv(S)
            Si[t], logdet[t] = Sinv, np.linalg.slogdet(S)[1]
            K[t] = P @ self.C.T @ Sinv
            Pf[t] = P - K[t] @ self.C @ P
            Pf[t] = (Pf[t] + Pf[t].T) / 2
        return Pp, Pf, K, Si, logdet

    def _filter_means(self, X, L, cov):
        Pp, Pf, K, Si, logdet = cov
        N, T, D = X.shape
        mp = np.empty((N, T, self.d))
        mf = np.empty((N, T, self.d))
        ll = np.zeros(N)
        m = np.broadcast_to(self.mu0, (N, self.d))
        for t in range(T):
            pred = m if t == 0 else mf[:, t - 1] @ self.A.T
            mp[:, t] = pred
            e = X[:, t] - pred @ self.C.T - self.b
            mf[:, t] = pred + e @ K[t].T
            alive = t < L
            ll += alive * -0.5 * (D * LOG2PI + logdet[t] + np.einsum("nd,de,ne->n", e, Si[t], e))
            m = mf[:, t]
        return mp, mf, ll

    def fit(self, seqs: list[np.ndarray]) -> "GaussianLDS":
        X, L = _pad(seqs)
        N, T, D = X.shape
        d = self.d
        rng = np.random.default_rng(self.random_state)
        self._init(X, L, rng)
        mask = np.arange(T)[None] < L[:, None]
        n_obs = mask.sum()
        lengths, counts = np.unique(L, return_counts=True)
        self.history_ = []
        for _ in range(self.n_iter):
            cov = self._cov_pass(T)
            Pp, Pf = cov[0], cov[1]
            mp, mf, ll = self._filter_means(X, L, cov)
            self.history_.append(float(ll.sum() / n_obs))
            J = np.stack([Pf[t] @ self.A.T @ np.linalg.inv(Pp[t + 1]) for t in range(T - 1)]) \
                if T > 1 else np.zeros((0, d, d))
            ms = mf.copy()
            for t in range(T - 2, -1, -1):
                upd = mf[:, t] + (ms[:, t + 1] - mp[:, t + 1]) @ J[t].T
                ms[:, t] = np.where((t < L - 1)[:, None], upd, mf[:, t])
            sum_P = np.zeros((d, d))
            sum_first = np.zeros((d, d))
            sum_last = np.zeros((d, d))
            sum_lag = np.zeros((d, d))
            for Ln, c in zip(lengths, counts):
                Ps = Pf[Ln - 1]
                sum_last += c * Ps
                sum_P += c * Ps
                for t in range(Ln - 2, -1, -1):
                    sum_lag += c * (Ps @ J[t].T)
                    Ps = Pf[t] + J[t] @ (Ps - Pp[t + 1]) @ J[t].T
                    sum_P += c * Ps
                sum_first += c * Ps
            m_all, Y = ms[mask], X[mask]
            Ezz = sum_P + m_all.T @ m_all
            Sz, Sy, Syz = m_all.sum(0), Y.sum(0), Y.T @ m_all
            pm = mask[:, 1:]
            prev, nxt = ms[:, :-1][pm], ms[:, 1:][pm]
            n_tr = pm.sum()
            Sprev = (sum_P - sum_last) + prev.T @ prev
            Snext = (sum_P - sum_first) + nxt.T @ nxt
            Slag = sum_lag + nxt.T @ prev
            m0 = ms[:, 0]
            self.mu0 = m0.mean(0)
            dev = m0 - self.mu0
            self.V0 = sum_first / N + dev.T @ dev / N + 1e-6 * np.eye(d)
            self.A = Slag @ np.linalg.inv(Sprev)
            Q = (Snext - self.A @ Slag.T) / max(n_tr, 1)
            self.Q = (Q + Q.T) / 2 + 1e-6 * np.eye(d)
            G = np.block([[Ezz, Sz[:, None]], [Sz[None], np.array([[n_obs]])]])
            Cb = np.hstack([Syz, Sy[:, None]]) @ np.linalg.inv(G)
            self.C, self.b = Cb[:, :d], Cb[:, d]
            Syy = (Y ** 2).sum(0)
            self.R = np.maximum((Syy - (Cb * np.hstack([Syz, Sy[:, None]])).sum(1)) / n_obs,
                                self.r_floor)
            if len(self.history_) > 1 and abs(self.history_[-1] - self.history_[-2]) < \
                    self.tol * abs(self.history_[-2]):
                break
        return self

    def filter(self, seqs: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        X, L = _pad(seqs)
        _, mf, ll = self._filter_means(X, L, self._cov_pass(X.shape[1]))
        return [mf[i, : L[i]] for i in range(len(seqs))], ll
