import itertools

import numpy as np
import pytest

from hmm_agents.hmm import CategoricalHMM


def random_model(K=3, M=4, seed=0):
    rng = np.random.default_rng(seed)
    m = CategoricalHMM(n_states=K, n_symbols=M)
    return m.set_params(rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K), K),
                        rng.dirichlet(np.ones(M), K))


def sample(m, n, lo=2, hi=12, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        T = rng.integers(lo, hi)
        z = rng.choice(m.n_states, p=m.startprob_)
        xs = []
        for _ in range(T):
            xs.append(rng.choice(m.n_symbols, p=m.emissionprob_[z]))
            z = rng.choice(m.n_states, p=m.transmat_[z])
        out.append(np.array(xs))
    return out


def joint(m, x, z):
    p = m.startprob_[z[0]] * m.emissionprob_[z[0], x[0]]
    for t in range(1, len(x)):
        p *= m.transmat_[z[t - 1], z[t]] * m.emissionprob_[z[t], x[t]]
    return p


def test_forward_matches_brute_force():
    m = random_model()
    for x in sample(m, 5, hi=6):
        brute = sum(joint(m, x, z) for z in itertools.product(range(3), repeat=len(x)))
        assert np.isclose(m.score([x]), np.log(brute))


def test_variable_lengths_batch_equals_individual():
    m = random_model()
    seqs = sample(m, 20)
    batch = m.score_samples(seqs)
    single = [m.score_samples([s])[0] for s in seqs]
    assert np.allclose(batch, single)


def test_viterbi_matches_brute_force():
    m = random_model()
    for x in sample(m, 5, hi=6):
        best = max(itertools.product(range(3), repeat=len(x)), key=lambda z: joint(m, x, z))
        assert tuple(m.viterbi([x])[0]) == best


def test_filter_is_posterior_given_prefix_only():
    m = random_model()
    x = sample(m, 1, lo=6, hi=7)[0]
    filt = m.filter([x])[0]
    for t in range(len(x)):
        post = np.zeros(3)
        for z in itertools.product(range(3), repeat=t + 1):
            post[z[-1]] += joint(m, x[: t + 1], z)
        assert np.allclose(filt[t], post / post.sum())


def test_smoothed_posteriors_sum_to_one():
    m = random_model()
    for g in m.smooth(sample(m, 10)):
        assert np.allclose(g.sum(1), 1)


def test_matches_hmmlearn():
    hmmlearn = pytest.importorskip("hmmlearn.hmm")
    m = random_model()
    ref = hmmlearn.CategoricalHMM(n_components=3)
    ref.startprob_, ref.transmat_, ref.emissionprob_ = m.startprob_, m.transmat_, m.emissionprob_
    ref.n_features = 4
    seqs = sample(m, 30)
    X, lengths = np.concatenate(seqs)[:, None], [len(s) for s in seqs]
    assert np.isclose(m.score(seqs), ref.score(X, lengths))


def test_em_increases_likelihood_and_recovers_structure():
    true = CategoricalHMM(n_states=2, n_symbols=3).set_params(
        np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.2, 0.8]]),
        np.array([[0.8, 0.15, 0.05], [0.05, 0.15, 0.8]]))
    data = sample(true, 300, lo=20, hi=40)
    m = CategoricalHMM(n_states=2, n_symbols=3, n_iter=200, n_init=2, tol=1e-6).fit(data)
    h = np.array(m.history_)
    assert (np.diff(h) > -1e-6).all(), "EM log-likelihood must be non-decreasing"
    order = np.argsort(m.emissionprob_[:, 0])[::-1]
    assert np.allclose(m.emissionprob_[order], true.emissionprob_, atol=0.06)
    assert np.allclose(m.transmat_[np.ix_(order, order)], true.transmat_, atol=0.06)


def test_save_load_roundtrip(tmp_path):
    m = random_model()
    m.save(tmp_path / "m.npz")
    m2 = CategoricalHMM.load(tmp_path / "m.npz")
    seqs = sample(m, 5)
    assert np.isclose(m.score(seqs), m2.score(seqs))
