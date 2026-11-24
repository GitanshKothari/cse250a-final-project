import numpy as np
from scipy.stats import multivariate_normal

from hmm_agents.lds import GaussianLDS


def _data(n=200, d=2, D=5, seed=0):
    r = np.random.default_rng(seed)
    A = np.array([[0.95, 0.1], [0, 0.8]])
    C = r.normal(size=(D, d))
    seqs = []
    for _ in range(n):
        z, ys = r.normal(size=d), []
        for _ in range(r.integers(5, 30)):
            ys.append(C @ z + 0.3 * r.normal(size=D))
            z = A @ z + 0.3 * r.normal(size=d)
        seqs.append(np.array(ys))
    return seqs


def test_em_monotone_and_exact_likelihood():
    seqs = _data()
    m = GaussianLDS(d=2, n_iter=25, tol=0).fit(seqs)
    assert (np.diff(m.history_) > -1e-8).all()
    y = seqs[0][:5]
    T, D = y.shape
    Ez, Pz = [m.mu0], [m.V0]
    for _ in range(1, T):
        Pz.append(m.A @ Pz[-1] @ m.A.T + m.Q)
        Ez.append(m.A @ Ez[-1])
    mu = np.concatenate([m.C @ e + m.b for e in Ez])
    Sig = np.zeros((T * D, T * D))
    for i in range(T):
        for j in range(T):
            cz = (np.linalg.matrix_power(m.A, i - j) @ Pz[j] if i >= j
                  else (np.linalg.matrix_power(m.A, j - i) @ Pz[i]).T)
            Sig[i * D:(i + 1) * D, j * D:(j + 1) * D] = m.C @ cz @ m.C.T + \
                (np.diag(m.R) if i == j else 0)
    exact = multivariate_normal(mu, Sig).logpdf(y.ravel())
    assert abs(m.filter([y])[1][0] - exact) < 1e-6


def test_filter_is_causal():
    seqs = _data(50)
    m = GaussianLDS(d=2, n_iter=5).fit(seqs)
    full, _ = m.filter([seqs[0]])
    pre, _ = m.filter([seqs[0][:4]])
    assert np.allclose(full[0][:4], pre[0])
