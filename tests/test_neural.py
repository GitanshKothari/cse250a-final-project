import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import hmm_agents.models.feature_lr as fl
from hmm_agents.features import clear_feature_cache
from hmm_agents.models import FailurePredictor, build_model, prefix_view
from hmm_agents.runner import check_prefix_invariance

ACTS = ["inspect", "edit", "test", "submit", "other", "idle"]
FAST = {"hidden": 8, "max_epochs": 6, "patience": 6, "batch_size": 64, "n_threads": 1}
TINY_HMM = {"K": 2, "n_init": 1, "n_iter": 3, "max_train_runs": None}
TINY_LDS = {"d": 2, "n_iter": 3, "max_train_runs": None}


def _runs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        L = int(rng.integers(6, 30))
        fail = rng.random() < 0.7
        acts = rng.choice(ACTS, L)
        p_err = np.where(np.arange(L) >= 2, 0.7 if fail else 0.15, 0.3)
        errs = rng.random(L) < p_err
        rows.append({
            "run_id": i, "instance_id": f"o__r-{i // 4}", "model_name": "m" + str(i % 2),
            "target": not fail, "exit_status": "x", "issue_len": 100 + i,
            "n_steps": L, "split": "clf_train" if i % 2 else "hmm_train",
            "symbols": [ACTS.index(a) * 2 + int(e) for a, e in zip(acts, errs)],
            "actions": list(acts), "errors": list(errs),
            "commands": [f"c{rng.integers(5)}" for _ in range(L)],
            "viewed_files": [""] * L, "edited_files": [""] * L,
            "test_passed": [False] * L, "no_match": [False] * L, "success_msg": [False] * L})
    return pd.DataFrame(rows)


def _fit(spec, r, prefixes=(3, 5)):
    m = build_model({"type": "neural", **FAST, **spec})
    return m.fit(r.drop(columns="target"), ~r.target.to_numpy(), list(prefixes))


@pytest.mark.parametrize("spec", [
    {"name": "gru", "arch": "gru"},
    {"name": "mlp", "arch": "mlp"},
    {"name": "cnn", "arch": "cnn"},
    {"name": "cnn_w3", "arch": "cnn", "dilations": [1]},
    {"name": "gru+hmm+lds", "arch": "gru", "inputs": ["hmm", "lds"],
     "hmm": TINY_HMM, "lds": TINY_LDS},
])
def test_neural_variants_are_causal(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    r = _runs(200)
    m = _fit(spec, r)
    assert m.stochastic
    for t in (3, 5):
        check_prefix_invariance(m, r, t)
        p = m.predict_proba(prefix_view(r[r.n_steps > t], t), t)
        assert ((p > 0) & (p < 1)).all()
    with pytest.raises(ValueError, match="prefixes up to"):
        m.predict_proba(prefix_view(r, 6), 6)
    cfg = m.get_config()
    assert cfg["n_params"] > 0 and cfg["best_epoch"] >= 1
    assert cfg["inputs"][0] == "steps"


def test_gru_learns_planted_signal(tmp_path, monkeypatch):
    from sklearn.metrics import roc_auc_score
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    tr, te = _runs(600, seed=1), _runs(300, seed=2)
    m = _fit({"name": "gru", "arch": "gru", "max_epochs": 15, "patience": 15}, tr, (3, 10))
    alive = te[te.n_steps > 10]
    auc = roc_auc_score(~alive.target.to_numpy(), m.predict_proba(prefix_view(alive, 10), 10))
    assert auc > 0.8
    assert m.history_[m.best_epoch_ - 1] == min(m.history_)


def test_neural_shares_hmm_and_lds_cache_with_feature_lr(tmp_path, monkeypatch):
    clear_feature_cache()
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    r = _runs(150)
    lr = build_model({"name": "ref", "type": "feature_lr", "groups": ["hmm", "lds"],
                      "hmm": TINY_HMM, "lds": TINY_LDS, "hmm_data": "all"})
    lr.fit(r.drop(columns="target"), ~r.target.to_numpy(), [5])
    cache = tmp_path / "models" / "cache"
    before = sorted(p.name for p in cache.iterdir())
    nn_model = _fit({"name": "g", "inputs": ["hmm", "lds"], "hmm": TINY_HMM, "lds": TINY_LDS}, r, [5])
    assert sorted(p.name for p in cache.iterdir()) == before
    assert np.allclose(nn_model.hmm_.transmat_, lr.hmm_.transmat_)
    assert np.allclose(nn_model.lds_.A, lr.lds_.A)


def test_neural_save_load_and_seed_determinism(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    r = _runs(150)
    a, b = _fit({"name": "g"}, r), _fit({"name": "g"}, r)
    pv = prefix_view(r[r.n_steps > 5], 5)
    assert np.allclose(a.predict_proba(pv, 5), b.predict_proba(pv, 5))
    a.save(tmp_path / "g.pkl")
    assert np.allclose(FailurePredictor.load(tmp_path / "g.pkl").predict_proba(pv, 5),
                       a.predict_proba(pv, 5))


def test_cnn_receptive_field(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    r = _runs(200)
    short = _fit({"name": "cnn_w7", "arch": "cnn", "dilations": [1, 1, 1]}, r, (3, 10))
    full = _fit({"name": "cnn", "arch": "cnn"}, r, (3, 10))
    assert short.receptive_field == 7 and full.receptive_field == 31
    assert short.get_config()["receptive_field"] == 7
    alive = r[r.n_steps > 10].reset_index(drop=True)
    changed = alive.copy()
    changed["symbols"] = [[(s[0] + 2) % 12, *s[1:]] for s in changed["symbols"]]
    for m, same in ((short, True), (full, False)):
        a = m.predict_proba(prefix_view(alive, 10), 10)
        b = m.predict_proba(prefix_view(changed, 10), 10)
        assert np.allclose(a, b, atol=1e-12) == same, m.name


def test_neural_rejects_bad_specs():
    with pytest.raises(ValueError, match="apply to arch 'cnn' only"):
        build_model({"name": "x", "type": "neural", "arch": "gru", "dilations": [1]})
    with pytest.raises(ValueError, match="arch"):
        build_model({"name": "x", "type": "neural", "arch": "transformer"})
    with pytest.raises(ValueError, match="unknown inputs"):
        build_model({"name": "x", "type": "neural", "inputs": ["ngram"]})
