import numpy as np

from hmm_agents.features import (hmm_memory_features, markov_features, sticky_states,
                                 MARKOV_FEATURES)
from hmm_agents.hmm import CategoricalHMM
from hmm_agents.markov import MarkovChain


def small_hmm():
    return CategoricalHMM(n_states=3, n_symbols=10).set_params(
        np.array([0.5, 0.3, 0.2]),
        np.array([[0.95, 0.04, 0.01], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]),
        np.random.default_rng(0).dirichlet(np.ones(10), 3))


def test_markov_chain_score_matches_hand_computation():
    seqs = [np.array([0, 1, 1, 2]), np.array([1, 2, 0])]
    mc = MarkovChain(n_symbols=3, pseudocount=0.0).fit(seqs)
    assert np.allclose(mc.startprob_, [0.5, 0.5, 0.0])
    assert np.allclose(mc.transmat_[1], [0.0, 1 / 3, 2 / 3])
    expected = np.log(0.5) + np.log(mc.transmat_[0, 1]) + np.log(1 / 3) + np.log(2 / 3)
    assert np.isclose(mc.score_samples([seqs[0]])[0], expected)


def test_markov_save_load_roundtrip(tmp_path):
    mc = MarkovChain(n_symbols=10).fit([np.array([0, 3, 3, 5]), np.array([2, 2])])
    mc.save(tmp_path / "m.npz")
    mc2 = MarkovChain.load(tmp_path / "m.npz")
    s = [np.array([0, 3, 5])]
    assert np.isclose(mc.score(s), mc2.score(s))


def test_sticky_states_from_transition_matrix_only():
    assert list(sticky_states(small_hmm())) == [0]


def test_prefix_features_ignore_steps_after_t():
    hmm, t = small_hmm(), 5
    mc = MarkovChain(n_symbols=10).fit([np.arange(10), np.arange(10)[::-1]])
    a = np.array([1, 4, 4, 2, 7, 0, 0, 0, 9])
    b = a.copy()
    b[t:] = [3, 8, 5, 6]
    assert np.allclose(hmm_memory_features(hmm, [a], t), hmm_memory_features(hmm, [b], t))
    assert np.allclose(markov_features(mc, [a], t), markov_features(mc, [b], t))


def test_feature_shapes_and_meaning():
    hmm = small_hmm()
    mc = MarkovChain(n_symbols=10).fit([np.arange(10)])
    seq = np.array([1, 1, 1, 2])
    mem = hmm_memory_features(hmm, [seq], 4)[0]
    assert np.isclose(mem[:3].sum(), 1)
    assert 0 <= mem[3] <= 1
    mk = markov_features(mc, [seq], 4)[0]
    assert len(mk) == len(MARKOV_FEATURES)
    assert mk[2] == 1
    assert np.isclose(mk[10 + 1], 0.75)
    assert np.isclose(mk[-1], 2 / 3)


def test_within_group_pair_counts_match_brute_force():
    from hmm_agents.evaluate import concordant, pair_counts
    rng = np.random.default_rng(1)
    n = 60
    P = rng.integers(0, 5, size=(n, 2)).astype(float)
    y = rng.random(n) < 0.6
    issue = rng.integers(0, 6, n)
    group = issue * 2 + rng.integers(0, 2, n)
    C, N = pair_counts(P, y, group, issue)
    for i in range(issue.max() + 1):
        pairs = [(a, b) for a in range(n) for b in range(n)
                 if y[a] and not y[b] and group[a] == group[b] and issue[a] == i]
        assert N[i] == len(pairs)
        for j in range(2):
            brute = sum((P[a, j] > P[b, j]) + 0.5 * (P[a, j] == P[b, j]) for a, b in pairs)
            assert C[i, j] == brute
    assert concordant(np.array([1.0, 2.0]), np.array([1.0, 0.0, 3.0])) == 1.5 + 2.0
