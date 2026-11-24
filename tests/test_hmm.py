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


def test_restart_lls_recorded_and_saved(tmp_path):
    rng = np.random.default_rng(1)
    seqs = [rng.integers(0, 4, rng.integers(5, 20)) for _ in range(40)]
    m = CategoricalHMM(n_states=3, n_symbols=4, n_iter=20, n_init=4, random_state=0).fit(seqs)
    assert len(m.restart_ll_) == len(m.restart_iters_) == 4
    total = sum(map(len, seqs))
    assert max(m.restart_ll_) == pytest.approx(m.score(seqs) / total)
    m.save(tmp_path / "m.npz")
    assert CategoricalHMM.load(tmp_path / "m.npz").restart_ll_ == pytest.approx(m.restart_ll_)


def test_history_features_are_causal_and_correct():
    from hmm_agents.features import (history_from_filtered, hmm_history_features,
                                     hmm_history_names)
    rng = np.random.default_rng(2)
    seqs = [rng.integers(0, 4, 25) for _ in range(30)]
    m = CategoricalHMM(n_states=3, n_symbols=4, n_iter=10, n_init=1).fit(seqs)
    full = m.filter(seqs)
    pref = m.filter([s[:10] for s in seqs])
    assert np.allclose(hmm_history_features(full, 10), hmm_history_features(pref, 10))
    assert hmm_history_features(full, 10).shape == (30, len(hmm_history_names(3)))
    f = np.eye(3)[[0, 0, 1, 1, 1, 2]] * 0.8 + 0.2 / 3
    h = history_from_filtered(f)
    occ, mx, ent, dwell, sw = h[:3], h[3:6], h[6:9], h[9], h[10]
    assert np.allclose(occ, f.mean(0)) and np.allclose(mx, f.max(0))
    assert list(ent) == [0, 1, 0] and dwell == 1 and sw == pytest.approx(2 / 6)


def test_phase_hmm_init_from_plain_hmm_matches_its_likelihood():
    from hmm_agents.hmm import PhaseHMM
    r = np.random.default_rng(2)
    seqs = [r.integers(0, 6, r.integers(3, 25)) for _ in range(150)]
    base = CategoricalHMM(3, 6, n_iter=20, n_init=1, random_state=0).fit(seqs)
    pi, A, B = PhaseHMM.params_from_hmm(base, 1)
    assert np.allclose(A, base.transmat_) and np.allclose(B, base.emissionprob_)
    ph = PhaseHMM(3, 2, 6, n_iter=15, n_init=1)
    ph.init_params = PhaseHMM.params_from_hmm(base, 2)
    assert (ph.init_params[1][~ph.trans_mask_] == 0).all()
    ph.fit(seqs)
    assert (np.diff(ph.history_) > -1e-9).all() and len(ph.history_) > 1
