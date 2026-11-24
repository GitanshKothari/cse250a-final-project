from collections import Counter

import pandas as pd
import pytest

from hmm_agents.config import ROOT
from hmm_agents.splits import (SPLIT_NAMES, assign_splits, build_split_table, hash_split,
                               load_split_table, split_fractions)

FRAC = dict(test=0.15, val=0.15, hmm_frac_of_train=0.5)


def _meta(n_issues=400, runs=5):
    return pd.DataFrame({
        "instance_id": [f"own__repo{k % 7}-{k}" for k in range(n_issues) for _ in range(runs)],
        "target": [(k * 31 + r) % 6 == 0 for k in range(n_issues) for r in range(runs)],
    })


def test_build_is_deterministic_and_order_independent():
    m = _meta()
    a = build_split_table(m, **FRAC, seed=0)
    b = build_split_table(m.sample(frac=1, random_state=1), **FRAC, seed=0)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.split) == set(SPLIT_NAMES)
    assert a.instance_id.is_unique


def test_build_proportions():
    tab = build_split_table(_meta(2000), **FRAC, seed=0)
    share = tab.split.value_counts(normalize=True)
    for name, f in split_fractions(**FRAC).items():
        assert share[name] == pytest.approx(f, abs=0.02)


def test_lookup_ignores_which_runs_are_present():
    m = _meta()
    frozen = build_split_table(m, **FRAC, seed=0).set_index("instance_id")["split"]
    sub = m[m.index % 3 != 0].reset_index(drop=True)
    split, n_fb = assign_splits(sub["instance_id"], frozen, split_fractions(**FRAC))
    assert n_fb == 0
    assert (split.values == sub["instance_id"].map(frozen).values).all()
    assert pd.DataFrame({"i": sub.instance_id, "s": split}).groupby("i").s.nunique().max() == 1


def test_hash_fallback_is_stable_and_proportional():
    fr = split_fractions(**FRAC)
    assert hash_split("new__issue-1", fr) == hash_split("new__issue-1", fr)
    c = Counter(hash_split(f"x__y-{k}", fr) for k in range(20000))
    for name, f in fr.items():
        assert c[name] / 20000 == pytest.approx(f, abs=0.015)
    ids = pd.Series(["own__repo0-0", "brand__new-9", "brand__new-9"])
    frozen = pd.Series({"own__repo0-0": "val"})
    split, n_fb = assign_splits(ids, frozen, fr)
    assert n_fb == 1 and split[0] == "val" and split[1] == split[2] == hash_split("brand__new-9", fr)


def test_committed_table_is_valid():
    path = ROOT / "configs" / "splits.csv"
    if not path.exists():
        pytest.skip("frozen table not built yet")
    s = load_split_table(path)
    assert len(s) == 3591
    assert set(s) == set(SPLIT_NAMES)
