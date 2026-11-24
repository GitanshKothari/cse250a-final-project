import argparse
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hmm_agents.config import load_config
from hmm_agents.parsing import ACTIONS

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--prefixes", default="5,10,20")
ap.add_argument("--n-boot", type=int, default=200)
args = ap.parse_args()
cfg = load_config(args.config)
PREFIXES = [int(x) for x in args.prefixes.split(",")]
rng = np.random.default_rng(cfg["seed"])
out_dir = cfg["paths"]["results"] / "experiments"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(cfg["paths"]["processed"] / "runs.parquet")
df = df[df.split.isin(["hmm_train", "clf_train"])].reset_index(drop=True)
print(f"runs: {len(df):,}  instances: {df.instance_id.nunique():,}  "
      f"success rate: {df.target.mean():.3f}")

def step_kind(a, e, nm):
    if a == "inspect":
        return "inspect:no_match" if nm else ("inspect:err" if e else "inspect:ok")
    return f"{a}:{'err' if e else 'ok'}"


nxt: dict[str, Counter] = {}
for acts, errs, nms in zip(df.actions, df.errors, df.no_match):
    kinds = [step_kind(a, e, n) for a, e, n in zip(acts, errs, nms)]
    for k, a_next in zip(kinds[:-1], acts[1:]):
        nxt.setdefault(k, Counter())[a_next] += 1


def dist(k):
    c = nxt.get(k, Counter())
    v = np.array([c[a] for a in ACTIONS], float)
    return v / v.sum()


beh = pd.DataFrame({k: dist(k) for k in ["inspect:ok", "inspect:err", "inspect:no_match"]},
                   index=[f"next={a}" for a in ACTIONS]).T
beh["n"] = [sum(nxt[k].values()) for k in beh.index]
js = {f"JS(no_match, {k})": jensenshannon(dist("inspect:no_match"), dist(k), base=2)
      for k in ["inspect:ok", "inspect:err"]}
js["JS(inspect:ok, inspect:err)"] = jensenshannon(dist("inspect:ok"), dist("inspect:err"), base=2)

def test_kind(a, e, tp, sm):
    if a != "test":
        return None
    if e:
        return "test:err"
    return "test:suite_pass" if tp else ("test:success_msg" if sm else "test:clean_other")


nxt_t: dict[str, Counter] = {}
for acts, errs, tps, sms in zip(df.actions, df.errors, df.test_passed, df.success_msg):
    for i in range(len(acts) - 1):
        k = test_kind(acts[i], errs[i], tps[i], sms[i])
        if k:
            nxt_t.setdefault(k, Counter())[acts[i + 1]] += 1
beh_t = pd.DataFrame({k: (lambda c: np.array([c[a] for a in ACTIONS], float) / sum(c.values()))(c)
                      for k, c in nxt_t.items()}, index=[f"next={a}" for a in ACTIONS]).T
beh_t["n"] = [sum(nxt_t[k].values()) for k in beh_t.index]

def features(row, t, nomatch_mode, pass_mode):
    acts, errs = row.actions[:t], np.array(row.errors[:t], bool)
    nm = np.array(row.no_match[:t], bool)
    if nomatch_mode == "ok" or nomatch_mode == "separate":
        errs = errs & ~nm
    streak = 0
    for e in errs[::-1]:
        if not e:
            break
        streak += 1
    tp = np.array(row.test_passed[:t], bool)
    if pass_mode == "msg":
        tp = tp | np.array(row.success_msg[:t], bool)
    elif pass_mode == "clean":
        tp = np.array([a == "test" for a in acts]) & ~np.array(row.errors[:t], bool)
    edits = [i for i, a in enumerate(acts) if a == "edit"]
    f = {
        "err_count": errs.sum(), "err_rate": errs.mean(), "err_streak": streak,
        "repeat_cmds": t - len(set(row.commands[:t])),
        "n_viewed_files": len({x for x in row.viewed_files[:t] if x}),
        "n_edited_files": len({x for x in row.edited_files[:t] if x}),
        "any_test_passed": float(tp.any()),
        "first_edit_step": edits[0] + 1 if edits else t + 1,
        "log_issue_len": np.log1p(row.issue_len),
        **{f"frac_{a}": sum(x == a for x in acts) / t for a in ACTIONS},
    }
    if nomatch_mode == "separate":
        f["no_match_count"] = nm.sum()
    return f


def cv_auroc(X, y, groups):
    scores = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
                                                                 class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return np.mean(scores), np.std(scores)


def boot_coef(X, y, groups, names, n_boot):
    uniq = np.unique(groups)
    idx_by_g = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array)
    coefs = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[idx], y[idx])
        coefs.append(clf[-1].coef_[0])
    c = np.array(coefs)
    return pd.DataFrame({"coef": c.mean(0), "lo": np.percentile(c, 2.5, 0),
                         "hi": np.percentile(c, 97.5, 0)}, index=names)


rows_pred, rows_coef, rows_pass = [], [], []
for t in PREFIXES:
    alive = df[df.n_steps > t].reset_index(drop=True)
    y = (~alive.target).to_numpy()
    g = alive.instance_id.to_numpy()

    cnt = np.array([[sum(1 for a, e, n in zip(r.actions[:t], r.errors[:t], r.no_match[:t])
                         if a == "inspect" and cond(e, n))
                     for cond in (lambda e, n: not e,
                                  lambda e, n: e and not n,
                                  lambda e, n: n)]
                    for r in alive.itertuples()], float)
    names = ["inspect_ok_count", "inspect_real_err_count", "inspect_no_match_count"]
    cf = boot_coef(cnt, y, g, names, args.n_boot).assign(t=t)
    rows_coef.append(cf)

    for nm_mode in ["err", "ok", "separate"]:
        for pass_mode in ["suite", "msg", "clean"]:
            if nm_mode != "err" and pass_mode != "suite":
                continue
            F = pd.DataFrame([features(r, t, nm_mode, pass_mode) for r in alive.itertuples()])
            m, s = cv_auroc(F.to_numpy(float), y, g)
            rows_pred.append({"t": t, "no_match_as": nm_mode, "test_passed_def": pass_mode,
                              "cv_auroc": m, "cv_sd": s, "n": len(alive)})
            if nm_mode == "err":
                tp = F["any_test_passed"].to_numpy() > 0
                rows_pass.append({
                    "t": t, "def": pass_mode, "prevalence": tp.mean(),
                    "success_rate_if_true": (~y[tp]).mean() if tp.any() else np.nan,
                    "success_rate_if_false": (~y[~tp]).mean(),
                    "univariate_auroc_success": roc_auc_score(~y, tp) if 0 < tp.mean() < 1 else np.nan,
                })

pred = pd.DataFrame(rows_pred)
coef = pd.concat(rows_coef)
passdef = pd.DataFrame(rows_pass)
beh.to_csv(out_dir / "exp01_next_action_after_inspect.csv")
beh_t.to_csv(out_dir / "exp01_next_action_after_test.csv")
pred.to_csv(out_dir / "exp01_cv_auroc.csv", index=False)
coef.to_csv(out_dir / "exp01_outcome_coefficients.csv")
passdef.to_csv(out_dir / "exp01_test_passed_definitions.csv", index=False)

pd.set_option("display.width", 160)
print("\n=== D1 behaviour: next action after each inspect outcome")
print(beh.round(3))
print({k: round(v, 3) for k, v in js.items()})
print("\n=== D1 outcome: standardised logistic coefficients (positive = more failure), 95% CI")
print(coef.round(3))
print("\n=== D3 behaviour: next action after each kind of test step")
print(beh_t.round(3))
print("\n=== D3 definitions of any_test_passed")
print(passdef.round(3))
print("\n=== Predictive value (5-fold issue-grouped CV AUROC, behaviour-feature LR)")
print(pred.pivot_table(index=["no_match_as", "test_passed_def"], columns="t",
                       values="cv_auroc").round(4))
print("(fold sd ~", round(pred.cv_sd.mean(), 4), ")")
