import numpy as np
import pytest

from hmm_agents.hmm import CategoricalHMM
from hmm_agents.state_match import (common_symbols, distinctness, hellinger,
                                    match_states, run_occupancy, state_shares)


def _random_stochastic(rng, shape):
    X = rng.random(shape) + 0.05
    return X / X.sum(-1, keepdims=True)


def test_hellinger_bounds_and_symmetry():
    rng = np.random.default_rng(0)
    P, Q = _random_stochastic(rng, (4, 6)), _random_stochastic(rng, (3, 6))
    H = hellinger(P, Q)
    assert H.shape == (4, 3) and (H >= 0).all() and (H <= 1 + 1e-12).all()
    assert np.allclose(hellinger(Q, P), H.T)
    assert np.allclose(np.diag(hellinger(P, P)), 0)
    assert np.isclose(hellinger([1, 0], [0, 1])[0, 0], 1.0)
    with pytest.raises(ValueError, match="symbol counts"):
        hellinger(P, Q[:, :5])


def test_common_symbols_renormalises_and_reports_dropped_mass():
    B = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    kept, dropped = common_symbols(B, 2)
    assert np.allclose(kept.sum(1), 1) and np.allclose(dropped, [0.2, 0.8])
    assert np.allclose(kept[0], [0.625, 0.375])
    with pytest.raises(ValueError, match="only dropped"):
        common_symbols(np.array([[0.0, 0.0, 1.0]]), 2)


def test_match_states_recovers_a_permutation():
    rng = np.random.default_rng(1)
    K, M = 6, 8
    A, B = _random_stochastic(rng, (K, K)), _random_stochastic(rng, (K, M)) ** 3
    B /= B.sum(1, keepdims=True)
    perm = rng.permutation(K)
    r, c, h, s = match_states(A, B, A[np.ix_(perm, perm)], B[perm])
    assert np.array_equal(perm[c], r) and np.allclose(h, 0) and np.allclose(s, 0)


def test_match_states_uses_stickiness_for_lookalike_states():
    B = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]) * 0.97 + 0.01
    A = np.array([[0.995, 0.004, 0.001], [0.3, 0.6, 0.1], [0.2, 0.2, 0.6]])
    swap = [1, 0, 2]
    r, c, _, s = match_states(A, B, A[np.ix_(swap, swap)], B[swap])
    assert dict(zip(r, c)) == {0: 1, 1: 0, 2: 2} and np.allclose(s, 0)


def test_match_states_across_symbol_counts():
    rng = np.random.default_rng(2)
    A, B10 = _random_stochastic(rng, (4, 4)), _random_stochastic(rng, (4, 10))
    B12 = np.hstack([B10 * 0.9, np.full((4, 2), 0.05)])
    r, c, h, _ = match_states(A, B10, A, B12)
    assert np.array_equal(r, c) and np.allclose(h, 0)


def _hmm(pi, A, B):
    B = np.asarray(B, float)
    return CategoricalHMM(n_states=len(pi), n_symbols=B.shape[1]).set_params(
        np.asarray(pi, float), np.asarray(A, float), B)


def test_distinctness_flags_rare_and_duplicate_states():
    hmm = _hmm([0.5, 0.49, 0.01],
               [[0.50, 0.49, 0.01], [0.44, 0.55, 0.01], [0.10, 0.10, 0.80]],
               [[0.90, 0.09, 0.01], [0.88, 0.11, 0.01], [0.01, 0.01, 0.98]])
    seqs = [np.array([0, 1, 0, 0, 1, 0]), np.array([1, 0, 0, 0])]
    d = distinctness(hmm, seqs)
    assert d["K"] == 3 and d["n_rare_modes"] >= 1 and d["smallest_mode_share"] == 0
    assert d["n_near_duplicate_pairs"] == 1 and d["closest_pair"] == "S0-S1"
    assert d["n_sticky"] == 0
    assert np.isclose(state_shares(hmm, seqs).sum(), 1)
    occ = run_occupancy(hmm, seqs)
    assert occ.shape == (2, 3) and np.allclose(occ.sum(1), 1)


def test_distinctness_clean_model():
    hmm = _hmm([0.5, 0.5], [[0.95, 0.05], [0.3, 0.7]], [[0.9, 0.1], [0.1, 0.9]])
    seqs = [np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])]
    d = distinctness(hmm, seqs)
    assert d["n_rare_modes"] == 0 and d["n_near_duplicate_pairs"] == 0 and d["n_sticky"] == 1
