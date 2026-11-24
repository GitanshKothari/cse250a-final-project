import copy

import numpy as np
import pandas as pd
import pytest
import yaml

import hmm_agents.models.feature_lr as fl
from hmm_agents.config import load_config
from hmm_agents.models import FailurePredictor, prefix_view
from hmm_agents.runner import rescore, run_experiment

ACTS = ["inspect", "edit", "test", "submit", "other", "idle"]


def _project(tmp_path):
    rng = np.random.default_rng(0)
    issues = [f"own__repo{k % 4}-{k}" for k in range(60)]
    split_of = {i: ["hmm_train", "clf_train", "clf_train", "hmm_train", "val"][k % 5]
                for k, i in enumerate(issues)}
    rows = []
    for k, iss in enumerate(issues):
        for r in range(6):
            L = int(rng.integers(6, 35))
            fail = rng.random() < 0.75
            errs = rng.random(L) < (0.55 if fail else 0.2)
            acts = rng.choice(ACTS, L)
            rows.append({"run_id": len(rows), "instance_id": iss, "model_name": "m",
                         "target": not fail, "exit_status": "x", "issue_len": 50 + k,
                         "n_steps": L, "split": split_of[iss],
                         "symbols": [ACTS.index(a) * 2 + int(e) for a, e in zip(acts, errs)],
                         "actions": list(acts), "errors": list(errs),
                         "commands": [f"c{rng.integers(6)}" for _ in range(L)],
                         "viewed_files": [""] * L, "edited_files": [""] * L,
                         "test_passed": [False] * L, "no_match": [False] * L,
                         "success_msg": [False] * L})
    runs = pd.DataFrame(rows)
    (tmp_path / "processed").mkdir()
    runs.to_parquet(tmp_path / "processed" / "runs.parquet", index=False)
    tab = runs.groupby("instance_id").agg(n_runs=("run_id", "size"),
                                          solve_rate=("target", "mean")).reset_index()
    tab["split"] = tab.instance_id.map(split_of)
    tab["repo"] = tab.instance_id.str.rsplit("-", n=1).str[0]
    tab.to_csv(tmp_path / "splits.csv", index=False)
    tr = tab[tab.split.isin(["hmm_train", "clf_train"])].sort_values("instance_id")
    pd.DataFrame({"instance_id": tr.instance_id, "fold": np.arange(len(tr)) % 3}).to_csv(
        tmp_path / "folds.csv", index=False)

    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(processed=tmp_path / "processed", results=tmp_path / "results",
                        logs=tmp_path / "logs")
    cfg["split"].update(file=str(tmp_path / "splits.csv"), folds_file=str(tmp_path / "folds.csv"))
    cfg["early"]["prefixes"] = [3, 5]
    cfg["eval"].update(n_boot=20, primary_t=5)
    return runs, cfg


EXP = {
    "name": "e2e",
    "claim": "synthetic end-to-end check",
    "seeds": [0, 1],
    "models": [
        {"name": "behavior", "type": "feature_lr", "groups": ["behavior", "context"],
         "grid": {"C": [0.01, 1.0]}},
        {"name": "hmm_hist", "type": "feature_lr", "groups": ["hmm", "hmm_hist", "context"],
         "hmm": {"K": 2, "n_init": 1, "n_iter": 3, "max_train_runs": None}},
    ],
    "paired": [["behavior", "hmm_hist"]],
}


def test_runner_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    runs, cfg = _project(tmp_path)
    res = tmp_path / "results" / "e2e"

    cv = run_experiment(copy.deepcopy(EXP), cfg, "cv")
    pred = pd.read_parquet(cv / "predictions.parquet")
    assert set(pred.method) == {"behavior", "hmm_hist@s0", "hmm_hist@s1"}
    assert "fold" in pred and pred.fold.notna().all()
    train_ids = set(runs[runs.split.isin(["hmm_train", "clf_train"])].run_id)
    assert set(pred.run_id) <= train_ids
    assert (pred.n_steps > pred.t).all()
    tuned = yaml.safe_load((res / "cv" / "tuning.yaml").read_text())
    assert tuned["behavior"]["C"] in (0.01, 1.0)
    assert len(pd.read_csv(cv / "tuning.csv")) == 2
    paired = pd.read_csv(cv / "paired.csv")
    assert {("behavior", "hmm_hist@s0"), ("behavior", "hmm_hist@s1")} <= set(zip(paired.a, paired.b))
    summ = pd.read_csv(cv / "seed_summary.csv")
    assert {"auroc", "d_auroc"} <= set(summ.kind) and (summ.n_seeds == 2).all()
    conf = yaml.safe_load((cv / "config.yaml").read_text())
    assert conf["leakage_check"] == "passed" and "runs_parquet_sha" in conf["provenance"]
    assert (cv / "auroc_vs_t.png").exists()
    assert not (tmp_path / "logs" / "val_runs.log").exists()
    runlog = (tmp_path / "logs" / "experiment_runs.log").read_text().splitlines()
    assert len(runlog) == 1 and "exp=e2e" in runlog[0] and "mode=cv" in runlog[0]
    assert "hmm_hist " in runlog[0] and "@s" not in runlog[0]

    val = run_experiment(copy.deepcopy(EXP), cfg, "val")
    vpred = pd.read_parquet(val / "predictions.parquet")
    assert set(vpred.run_id) <= set(runs[runs.split == "val"].run_id)
    assert yaml.safe_load((val / "config.yaml").read_text())["models"]["behavior"]["C"] == \
        tuned["behavior"]["C"]
    m = FailurePredictor.load(val / "models" / "hmm_hist@s1.pkl")
    held = runs[(runs.split == "val") & (runs.n_steps > 5)].reset_index(drop=True)
    want = vpred[(vpred.method == "hmm_hist@s1") & (vpred.t == 5)].set_index("run_id").p_fail
    assert np.allclose(m.predict_proba(prefix_view(held, 5), 5), want.reindex(held.run_id))
    log = (tmp_path / "logs" / "val_runs.log").read_text()
    assert "exp=e2e" in log and "synthetic end-to-end check" in log

    with pytest.raises(SystemExit, match="final"):
        run_experiment(copy.deepcopy(EXP), cfg, "test")
    with pytest.raises(SystemExit, match="claim"):
        run_experiment({**copy.deepcopy(EXP), "claim": None}, cfg, "val")

    before = pd.read_csv(cv / "metrics.csv")
    paired_before = pd.read_csv(cv / "paired.csv")
    assert len(paired_before) and paired_before["b"].str.contains("@s").any()
    rescore(copy.deepcopy(EXP), cfg, "cv", n_boot=20)
    pd.testing.assert_frame_equal(before, pd.read_csv(cv / "metrics.csv"))
    pd.testing.assert_frame_equal(paired_before, pd.read_csv(cv / "paired.csv"))
