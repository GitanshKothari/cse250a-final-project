import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from hmm_agents.evaluate import (compute_metrics, paired_diff, tradeoff, weighted_auroc,
                                 weighted_average_precision, weighted_brier)

rng = np.random.default_rng(0)


def _data(n=400, ties=False):
    y = rng.random(n) < 0.3
    s = rng.random(n) + 0.5 * y
    if ties:
        s = np.round(s, 1)
    return s, y.astype(float)


@pytest.mark.parametrize("ties", [False, True])
def test_unit_weights_match_sklearn(ties):
    s, y = _data(ties=ties)
    W = np.ones((1, len(s)))
    assert weighted_auroc(s, y, W)[0] == pytest.approx(roc_auc_score(y, s))
    assert weighted_average_precision(s, y, W)[0] == pytest.approx(average_precision_score(y, s))
    p = np.clip(s / s.max(), 0, 1)
    assert weighted_brier(p, y, W)[0] == pytest.approx(brier_score_loss(y, p))


@pytest.mark.parametrize("ties", [False, True])
def test_integer_weights_equal_repeated_rows(ties):
    s, y = _data(ties=ties)
    w = rng.integers(0, 4, len(s)).astype(float)
    rep = np.repeat(np.arange(len(s)), w.astype(int))
    W = np.stack([w, np.ones_like(w)])
    assert weighted_auroc(s, y, W)[0] == pytest.approx(roc_auc_score(y[rep], s[rep]))
    assert weighted_average_precision(s, y, W)[0] == pytest.approx(
        average_precision_score(y[rep], s[rep]))
    assert weighted_auroc(s, y, W)[1] == pytest.approx(roc_auc_score(y, s))


def _pred(n_issues=60, runs=8, t=5):
    rows = []
    for i in range(n_issues):
        for r in range(runs):
            fail = rng.random() < 0.8
            good = np.clip(0.8 + (0.15 if fail else -0.2) + 0.1 * rng.standard_normal(), 0, 1)
            rows += [("good", f"o__r{i % 9}-{i}", i * 100 + r, t, good, fail, t + 1 + r * 5),
                     ("noise", f"o__r{i % 9}-{i}", i * 100 + r, t, rng.random(), fail,
                      t + 1 + r * 5)]
    return pd.DataFrame(rows, columns=["method", "instance_id", "run_id", "t", "p_fail",
                                       "fail", "n_steps"])


def test_metrics_cis_and_cohorts():
    pred = _pred()
    m = compute_metrics(pred, n_boot=200, seen_repos={"o__r0", "o__r1"}, fixed_min_steps=20)
    a = m[(m.cohort == "alive")].set_index("method")
    assert a.loc["good", "auroc"] > 0.8 and abs(a.loc["noise", "auroc"] - 0.5) < 0.1
    assert (a.auroc_lo <= a.auroc).all() and (a.auroc <= a.auroc_hi).all()
    assert set(m.cohort) == {"alive", "fixed", "seen_repo", "unseen_repo"}
    fixed = m[(m.cohort == "fixed") & (m.method == "good")].iloc[0]
    assert fixed.n_runs == (pred[(pred.method == "good")].n_steps >= 20).sum()


def test_paired_diff():
    pred = _pred()
    same = paired_diff(pred, "good", "good", n_boot=100)
    assert np.allclose(same[["d_auroc", "d_auroc_lo", "d_auroc_hi"]].to_numpy(), 0)
    d = paired_diff(pred, "noise", "good", n_boot=200).set_index("cohort").loc["alive"]
    assert d.d_auroc > 0.2 and d.d_auroc_lo > 0 and d.p_b_better_auroc == 1.0
    assert d.d_brier < 0


def test_tradeoff_hand_example():
    pred = pd.DataFrame({"method": "m", "instance_id": list("abcd"), "run_id": range(4),
                         "t": 2, "p_fail": [0.9, 0.8, 0.2, 0.1],
                         "fail": [True, False, True, False], "n_steps": [12, 7, 5, 3]})
    r = tradeoff(pred, stop_fracs=(0.5,)).iloc[0]
    assert r.n_stopped == 2 and r.precision_fail == 0.5
    assert r.failures_caught == 0.5 and r.successes_lost == 0.5
    assert r.compute_saved == pytest.approx((10 + 5) / (10 + 5 + 3 + 1))


def test_rejects_leaky_predictions():
    pred = _pred()
    pred.loc[0, "n_steps"] = pred.loc[0, "t"]
    with pytest.raises(ValueError, match="hard rule 3"):
        compute_metrics(pred, n_boot=10)


def test_agent_cohorts_when_model_name_present():
    pred = _pred()
    pred["model_name"] = np.where(pred.run_id % 2, "a", "b")
    m = compute_metrics(pred, n_boot=20)
    assert {"agent=a", "agent=b"} <= set(m.cohort)
    d = paired_diff(pred, "noise", "good", n_boot=20)
    assert "agent=a" in set(d.cohort)


def test_within_fold_auroc_removes_fold_intercept_bias():
    n = 4000
    fold = np.repeat(np.arange(5), n // 5)
    rate = np.array([0.93, 0.95, 0.95, 0.96, 0.98])[fold]
    y = (rng.random(n) < rate).astype(float)
    s = 0.01 * rng.random(n) - 0.1 * rate
    W = np.ones((1, n))
    assert weighted_auroc(s, y, W)[0] < 0.45
    assert abs(weighted_auroc(s, y, W, fold)[0] - 0.5) < 0.05
    assert weighted_auroc(s, y, W, np.zeros(n))[0] == pytest.approx(weighted_auroc(s, y, W)[0])


def test_tradeoff_within_fold():
    pred = pd.DataFrame({"method": "m", "instance_id": list("abcd"), "run_id": range(4), "t": 1,
                         "p_fail": [0.9, 0.8, 0.2, 0.1], "fail": [True, False, True, False],
                         "n_steps": [5, 5, 5, 5], "fold": [0, 0, 1, 1]})
    r = tradeoff(pred, stop_fracs=(0.5,)).iloc[0]
    assert r.n_stopped == 2 and r.precision_fail == 1.0


def test_within_issue_auroc_brute_force_and_oracle():
    from hmm_agents.evaluate import (ORACLE_METHOD, issue_difficulty_oracle,
                                     weighted_auroc_within)
    r = np.random.default_rng(3)
    n = 400
    g = r.integers(0, 40, n)
    y = (r.random(n) < 0.3 + 0.01 * g).astype(float)
    s = np.round(r.random(n) + 0.5 * y, 1)
    w = r.integers(0, 3, (2, n)).astype(float)

    def brute(wv):
        num = den = 0.0
        for i in range(n):
            for j in range(n):
                if g[i] == g[j] and y[i] == 1 and y[j] == 0:
                    ww = wv[i] * wv[j]
                    den += ww
                    num += ww * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
        return num / den
    got = weighted_auroc_within(s, y, np.vstack([np.ones(n), w]), g)
    assert got[0] == pytest.approx(brute(np.ones(n)))
    assert got[1] == pytest.approx(brute(w[0])) and got[2] == pytest.approx(brute(w[1]))
    const = (g * 0.01).astype(float)
    assert weighted_auroc_within(const, y, np.ones((1, n)), g)[0] == pytest.approx(0.5)

    pred = pd.DataFrame({"method": "m", "run_id": np.arange(n), "instance_id": g.astype(str),
                         "t": 3, "p_fail": s / 1.5, "fail": y.astype(int), "n_steps": 10})
    o = issue_difficulty_oracle(pred)
    assert (o.method == ORACLE_METHOD).all() and len(o) == n
    m = compute_metrics(pd.concat([pred, o]), n_boot=20)
    row = m[(m.method == ORACLE_METHOD) & (m.cohort == "alive")].iloc[0]
    assert row.auroc_wi == pytest.approx(0.5) and row.auroc > 0.6
    assert (m.n_issues_mixed > 0).all()
