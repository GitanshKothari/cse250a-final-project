import numpy as np
import pandas as pd
import pytest

from hmm_agents.features import clear_feature_cache
from hmm_agents.models import FailurePredictor, build_model, prefix_view
from hmm_agents.runner import check_prefix_invariance
from hmm_agents.splits import build_folds

rng = np.random.default_rng(0)
ACTS = ["inspect", "edit", "test", "submit", "other", "idle"]


def _runs(n=300):
    rows = []
    for i in range(n):
        L = int(rng.integers(4, 40))
        acts = rng.choice(ACTS, L)
        errs = rng.random(L) < (0.5 if i % 3 else 0.2)
        rows.append({
            "run_id": i, "instance_id": f"o__r-{i // 5}", "model_name": "m" + str(i % 2),
            "target": bool(i % 3 == 0), "exit_status": "x", "issue_len": 100 + i,
            "n_steps": L, "split": "clf_train" if i % 2 else "hmm_train",
            "symbols": [ACTS.index(a) * 2 + int(e) for a, e in zip(acts, errs)],
            "actions": list(acts), "errors": list(errs),
            "commands": [f"c{rng.integers(5)}" for _ in range(L)],
            "viewed_files": [""] * L, "edited_files": [""] * L,
            "test_passed": [False] * L, "no_match": [False] * L, "success_msg": [False] * L})
    return pd.DataFrame(rows)


def test_prefix_view_hides_future_and_outcome():
    r = _runs(20).assign(secret=1.0)
    pv = prefix_view(r, 3)
    assert not {"target", "exit_status", "n_steps", "split", "secret"} & set(pv.columns)
    assert all(len(s) == 3 for s in pv["symbols"]) and (pv["t"] == 3).all()


def test_feature_lr_fits_predicts_and_passes_leakage_check():
    clear_feature_cache()
    r = _runs()
    m = build_model({"name": "beh", "type": "feature_lr", "groups": ["behavior", "context"]})
    m.fit(r.drop(columns="target"), ~r.target.to_numpy(), [3, 5])
    p = m.predict_proba(prefix_view(r[r.n_steps > 3], 3), 3)
    assert p.shape == ((r.n_steps > 3).sum(),) and ((0 <= p) & (p <= 1)).all()
    check_prefix_invariance(m, r, 3)


def test_feature_lr_with_hmm(tmp_path, monkeypatch):
    clear_feature_cache()
    import hmm_agents.models.feature_lr as fl
    r = _runs(200)
    m = fl.FeatureLR("h", ["hmm", "hmm_hist", "behavior", "context"],
                     hmm={"K": 3, "n_init": 1, "n_iter": 5, "max_train_runs": None})
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    m.fit(r.drop(columns="target"), ~r.target.to_numpy(), [5])
    from hmm_agents.features import BEHAVIOR_FEATURES
    assert len(m.columns()) == (3 + 1) + (3 * 3 + 2) + len(BEHAVIOR_FEATURES) + 1
    check_prefix_invariance(m, r, 5)
    assert list((tmp_path / "models" / "cache").glob("hmm_*.npz"))


def test_feature_lr_hmm_data_all(tmp_path, monkeypatch):
    clear_feature_cache()
    import hmm_agents.models.feature_lr as fl
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    r = _runs(200)
    seen, lr_rows = [], []
    real_hmm, real_pv = fl.fit_or_load_hmm, fl.prefix_view
    monkeypatch.setattr(fl, "fit_or_load_hmm",
                        lambda runs, *a, **k: seen.append(len(runs)) or real_hmm(runs, *a, **k))
    monkeypatch.setattr(fl, "prefix_view",
                        lambda runs, t: lr_rows.append(len(runs)) or real_pv(runs, t))
    hmm = {"K": 3, "n_init": 1, "n_iter": 5, "max_train_runs": None}
    y = ~r.target.to_numpy()
    fl.FeatureLR("s", ["hmm_hist", "behavior"], hmm=hmm).fit(r, y, [5])
    full = fl.FeatureLR("a", ["hmm_hist", "behavior"], hmm=hmm, hmm_data="all").fit(r, y, [5])
    alive = r.n_steps > 5
    assert seen == [(r.split == "hmm_train").sum(), len(r)]
    assert lr_rows == [(alive & (r.split == "clf_train")).sum(), alive.sum()]
    assert full.get_config()["hmm_data"] == "all"
    check_prefix_invariance(full, r, 5)
    with pytest.raises(ValueError):
        fl.FeatureLR("x", ["hmm"], hmm_data="both")


class _Cheater(FailurePredictor):
    name = "cheater"

    def fit(self, runs, y_fail, prefixes):
        return self

    def predict_proba(self, prefix_runs, t):
        x = np.array([sum(s) for s in prefix_runs["symbols"]], float)
        return 1 / (1 + np.exp(-(x - x.mean()) / (x.std() + 1e-9)))


def test_leakage_check_catches_batch_dependence():
    with pytest.raises(AssertionError, match="batch"):
        check_prefix_invariance(_Cheater(), _runs(), 5)


def test_folds_grouped_balanced_deterministic():
    tab = pd.DataFrame({"instance_id": [f"i{k}" for k in range(500)],
                        "split": ["hmm_train", "clf_train", "val", "test", "clf_train"] * 100,
                        "solve_rate": [(k % 7) / 7 for k in range(500)]})
    f1, f2 = build_folds(tab, 5, 0), build_folds(tab.sample(frac=1, random_state=3), 5, 0)
    pd.testing.assert_frame_equal(f1.reset_index(drop=True), f2.reset_index(drop=True))
    assert set(f1.instance_id) == set(tab[tab.split.isin(["hmm_train", "clf_train"])].instance_id)
    assert f1.fold.value_counts().max() - f1.fold.value_counts().min() <= 1


def test_ngram_features():
    from hmm_agents.features import ngram_features, ngram_names
    f = ngram_features([[0, 1, 1, 3]], 3)[0]
    assert len(f) == len(ngram_names()) == 12 + 144
    assert f[0] == pytest.approx(1 / 3) and f[1] == pytest.approx(2 / 3)
    assert f[12 + 0 * 12 + 1] == 0.5 and f[12 + 1 * 12 + 1] == 0.5
    clear_feature_cache()
    r = _runs()
    m = build_model({"name": "ng", "type": "feature_lr", "groups": ["ngram", "context"]})
    m.fit(r.drop(columns="target"), ~r.target.to_numpy(), [5])
    check_prefix_invariance(m, r, 5)


def test_trigram_features_and_selection():
    from hmm_agents.features import trigram_matrix, trigram_name
    m = trigram_matrix([[0, 1, 2, 0, 1], [3, 3]], 4)
    assert m.shape == (2, 12 ** 3) and m[1].nnz == 0
    assert m[0, 0 * 144 + 1 * 12 + 2] == 0.5 and m[0, 1 * 144 + 2 * 12 + 0] == 0.5
    assert trigram_name(1 * 144 + 2 * 12 + 0).startswith("tri_")
    clear_feature_cache()
    r = _runs()
    mod = build_model({"name": "tri", "type": "feature_lr",
                       "groups": ["ngram", "trigram", "context"], "trigram_min_frac": 0.02})
    mod.fit(r.drop(columns="target"), ~r.target.to_numpy(), [5, 10])
    assert 0 < len(mod.tri_cols_[10]) < 12 ** 3
    check_prefix_invariance(mod, r, 10)
    assert len(mod.coefficients()) == sum(len(mod.columns(t)) for t in (5, 10))



def test_group_holdout_is_by_issue():
    from hmm_agents.models import group_holdout
    r = _runs()
    fit, hold = group_holdout(r, 0.2, seed=0)
    assert (fit ^ hold).all() and 0.1 < hold.mean() < 0.3
    assert not set(r.instance_id[fit]) & set(r.instance_id[hold])
    assert (group_holdout(r, 0.2, seed=0)[1] == hold).all()


def test_model_save_load_roundtrip(tmp_path):
    from hmm_agents.models import FailurePredictor
    clear_feature_cache()
    r = _runs()
    m = build_model({"name": "beh", "type": "feature_lr", "groups": ["behavior", "context"]})
    m.fit(r.drop(columns="target"), ~r.target.to_numpy(), [5])
    m.save(tmp_path / "m.pkl")
    pv = prefix_view(r[r.n_steps > 5], 5)
    assert np.array_equal(FailurePredictor.load(tmp_path / "m.pkl").predict_proba(pv, 5),
                          m.predict_proba(pv, 5))


def test_stochastic_flag():
    assert not build_model({"name": "b", "type": "feature_lr", "groups": ["behavior"]}).stochastic
    assert build_model({"name": "h", "type": "feature_lr", "groups": ["hmm"]}).stochastic


def test_expand_grid_and_limit():
    from hmm_agents.runner import expand_grid
    assert expand_grid(None) == [{}]
    assert expand_grid({"C": [0.1, 1], "K": [4, 8]}) == [
        {"C": 0.1, "K": 4}, {"C": 0.1, "K": 8}, {"C": 1, "K": 4}, {"C": 1, "K": 8}]
    with pytest.raises(ValueError, match="20"):
        expand_grid({"a": list(range(5)), "b": list(range(5))})


def test_step_encoder():
    from hmm_agents.encoding import D, STEP_FEATURE_NAMES, encode_prefixes, encode_run, pad_batch
    assert D == len(STEP_FEATURE_NAMES) == 19
    row = {"symbols": [0, 3, 3, 4], "no_match": [0, 0, 1, 0], "success_msg": [0, 0, 0, 1],
           "test_passed": [0, 0, 0, 0], "commands": ["a", "b", "b", "a"],
           "viewed_files": ["x", "", "x", "y"], "edited_files": ["", "x", "x", ""]}
    x = encode_run(row)
    col = {n: i for i, n in enumerate(STEP_FEATURE_NAMES)}
    assert x.shape == (4, 19) and x[:, :12].sum(1).tolist() == [1, 1, 1, 1]
    assert x[:, col["repeat_cmd"]].tolist() == [0, 0, 1, 1]
    assert x[:, col["new_view"]].tolist() == [1, 0, 0, 1]
    assert x[:, col["new_edit"]].tolist() == [0, 1, 0, 0]
    assert np.allclose(encode_run(row, 2), x[:2])
    r = _runs(10)
    seqs = encode_prefixes(prefix_view(r, 4))
    assert all(np.allclose(s, encode_run(row_, 4)) for s, row_ in zip(seqs, r.to_dict("records")))
    X, mask = pad_batch([x, x[:2]])
    assert X.shape == (2, 4, 19) and mask.sum() == 6 and not X[1, 2:].any()



def test_hgb_and_landmark_variants_fit_and_are_causal():
    clear_feature_cache()
    r = _runs()
    y = ~r.target.to_numpy()
    for spec in [{"classifier": "hgb", "clf_params": {"max_iter": 20}},
                 {"landmark": True},
                 {"landmark": True, "classifier": "hgb", "clf_params": {"max_iter": 20}}]:
        m = build_model({"name": "x", "type": "feature_lr", "groups": ["behavior", "ngram"], **spec})
        m.fit(r.drop(columns="target"), y, [3, 5])
        assert not m.stochastic
        for t in (3, 5):
            p = m.predict_proba(prefix_view(r[r.n_steps > t], t), t)
            assert ((0 <= p) & (p <= 1)).all()
            check_prefix_invariance(m, r, t)
    with pytest.raises(ValueError):
        build_model({"name": "x", "type": "feature_lr", "groups": ["trigram"], "landmark": True})


def test_markov_context_counts_and_llr():
    from hmm_agents.models.generative import M, MarkovScorer, context_counts
    C = context_counts([[0, 1, 1], [2]], order=1)
    assert C[0, M * M + 0] == 1 and C[0, 0 * M + 1] == 1 and C[0, 1 * M + 1] == 1
    assert C[1, M * M + 2] == 1 and C.sum() == 4
    C2 = context_counts([[0, 1, 1]], order=2)
    assert C2.sum() == 3 and C2.shape[1] == (M + 1) ** 2 * M
    runs = pd.DataFrame({"symbols": [[0, 1] * 5] * 20 + [[2, 3] * 5] * 20, "n_steps": 10})
    y = np.array([True] * 20 + [False] * 20)
    sc = MarkovScorer(order=1).fit(runs, y, [4])
    f = sc.features(prefix_view(runs.assign(run_id=0), 4), 4).iloc[:, 0].to_numpy()
    assert (f[:20] > 0).all() and (f[20:] < 0).all()


def test_supervised_groups_crossfit_and_cchmm(tmp_path, monkeypatch):
    clear_feature_cache()
    import hmm_agents.models.feature_lr as fl
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    r = _runs(200)
    y = ~r.target.to_numpy()
    hmm = {"K": 3, "n_init": 1, "n_iter": 5, "max_train_runs": None}
    m = build_model({"name": "mk", "type": "feature_lr", "groups": ["markov", "behavior"],
                     "scorers": {"markov": {"order": 1}}, "crossfit_folds": 3})
    m.fit(r.drop(columns="target"), y, [5])
    assert not m.stochastic and "markov1_llr" in m.columns(5)
    check_prefix_invariance(m, r, 5)
    c = build_model({"name": "cc", "type": "feature_lr", "groups": ["cchmm"], "hmm": hmm,
                     "hmm_data": "all", "scorers": {"cchmm": {"n_iter": 3}},
                     "crossfit_folds": 2})
    c.fit(r.drop(columns="target"), y, [5])
    assert c.stochastic and c.columns(5) == ["cchmm_llr", "cchmm_llr_per_step"]
    sc = c.scorers_["cchmm"]
    assert np.allclose(sc.B_.sum(1), 1) and all(np.allclose(A.sum(1), 1) for A in sc.A_)
    assert sc.transition_contrast().shape == (3, 3)
    check_prefix_invariance(c, r, 5)
    with pytest.raises(ValueError):
        build_model({"name": "x", "type": "feature_lr", "groups": ["cchmm"],
                     "hmm": {"alphabet": "rich"}})


def test_rich_alphabet_and_phase_hmm(tmp_path, monkeypatch):
    clear_feature_cache()
    import hmm_agents.models.feature_lr as fl
    from hmm_agents.encoding import N_RICH, rich_symbols
    from hmm_agents.hmm import CategoricalHMM, PhaseHMM
    row = {"symbols": [0, 0, 5], "no_match": [0, 0, 0], "success_msg": [0, 0, 1],
           "test_passed": [0, 0, 0], "commands": ["a", "a", "b"],
           "viewed_files": ["f", "f", ""], "edited_files": ["", "", ""]}
    assert list(rich_symbols(row)) == [0 * 8 + 2, 0 * 8 + 4, 5 * 8 + 1]
    assert list(rich_symbols(row, 2)) == list(rich_symbols(row))[:2] and N_RICH == 96
    rng = np.random.default_rng(1)
    seqs = [rng.integers(0, 12, rng.integers(3, 30)) for _ in range(200)]
    a = CategoricalHMM(3, 12, n_iter=10, n_init=1, random_state=0).fit(seqs)
    b = PhaseHMM(3, 1, 12, n_iter=10, n_init=1, random_state=0).fit(seqs)
    assert np.allclose(a.transmat_, b.transmat_, atol=1e-8)
    h = PhaseHMM(3, 3, 12, n_iter=10, n_init=1, random_state=0).fit(seqs)
    assert (h.transmat_[~h.trans_mask_] == 0).all() and (h.startprob_[~h.start_mask_] == 0).all()
    assert np.allclose(h.emissionprob_[0], h.emissionprob_[2])
    assert np.allclose(h.dwell_pmf(400).sum(1), 1, atol=1e-3)
    assert h.filter_macro(seqs[:2])[0].shape == (len(seqs[0]), 3)
    h.save(tmp_path / "p.npz")
    assert np.allclose(PhaseHMM.load(tmp_path / "p.npz").transmat_, h.transmat_)
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    r = _runs(150)
    for hs in [{"alphabet": "rich"}, {"n_phases": 2}]:
        m = build_model({"name": "v", "type": "feature_lr", "groups": ["hmm", "hmm_hist"],
                         "hmm_data": "all",
                         "hmm": {"K": 3, "n_init": 1, "n_iter": 3, "max_train_runs": None, **hs}})
        m.fit(r.drop(columns="target"), ~r.target.to_numpy(), [5])
        assert len(m.columns(5)) == (3 + 1) + (3 * 3 + 2)
        check_prefix_invariance(m, r, 5)


def test_lds_group(tmp_path, monkeypatch):
    clear_feature_cache()
    import hmm_agents.models.feature_lr as fl
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    r = _runs(150)
    for data in ("all", "split"):
        m = build_model({"name": "k", "type": "feature_lr", "groups": ["lds", "behavior"],
                         "hmm_data": data, "lds": {"d": 2, "n_iter": 3, "max_train_runs": None}})
        assert m.stochastic
        m.fit(r.drop(columns="target"), ~r.target.to_numpy(), [5])
        assert [c for c in m.columns(5) if c.startswith("lds")] == \
            ["lds_m0", "lds_m1", "lds_avg0", "lds_avg1", "lds_ll"]
        check_prefix_invariance(m, r, 5)
    assert list((tmp_path / "models" / "cache").glob("lds_*.pkl"))


def test_cv_checkpoints_resume(tmp_path, monkeypatch):
    import hmm_agents.runner as rn
    clear_feature_cache()
    r = _runs(120)
    folds = pd.Series(np.arange(len(r)) // 5 % 3)
    calls = []
    real = rn._cv_predictions
    monkeypatch.setattr(rn, "_cv_predictions", lambda *a, **k: calls.append(a[4]) or real(*a, **k))
    spec = {"name": "beh", "type": "feature_lr", "groups": ["behavior"]}
    p1, c1 = rn._cv_predictions_ckpt(spec, r, folds, [3], 0, "beh", tmp_path, "data1")
    p2, c2 = rn._cv_predictions_ckpt(spec, r, folds, [3], 0, "beh", tmp_path, "data1")
    assert len(calls) == 1 and c1 == c2
    pd.testing.assert_frame_equal(pd.concat(p1, ignore_index=True), p2[0])
    rn._cv_predictions_ckpt({**spec, "C": 0.5}, r, folds, [3], 0, "beh", tmp_path, "data1")
    rn._cv_predictions_ckpt(spec, r, folds, [3], 0, "beh", tmp_path, "data2")
    assert len(calls) == 3


def test_val_mode_paths_for_every_final_model_type(tmp_path):
    clear_feature_cache()
    r = _runs()
    y = ~r.target.to_numpy()
    from hmm_agents.models import FailurePredictor
    for spec in [{"groups": ["behavior"]},
                 {"groups": ["ngram", "behavior"], "landmark": True, "classifier": "hgb",
                  "clf_params": {"max_iter": 10}},
                 {"groups": ["behavior"], "classifier": "hgb", "clf_params": {"max_iter": 10}}]:
        m = build_model({"name": "x", "type": "feature_lr", **spec}).fit(r, y, [5])
        c = m.coefficients()
        assert (c is None) == (spec.get("classifier") == "hgb")
        m.save(tmp_path / "m.pkl")
        p = prefix_view(r[r.n_steps > 5], 5)
        assert np.allclose(FailurePredictor.load(tmp_path / "m.pkl").predict_proba(p, 5),
                           m.predict_proba(p, 5))
