import argparse

from hmm_agents.config import load_config
from hmm_agents.data import N_SHARDS, build_runs_table, stratified_subsample
from hmm_agents.parsing import SYMBOLS
from hmm_agents.splits import make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)

shards = cfg["data"]["shards"]
shards = range(N_SHARDS) if shards is None else shards
raw_dir = cfg["paths"]["raw"] / "data"
raw = sorted(f for s in shards for f in raw_dir.glob(f"train-{s:05d}-*.parquet"))
if not raw:
    raise SystemExit("No raw shards found. Run scripts/01_download.py first.")

df = build_runs_table(raw, min_steps=cfg["data"]["min_steps"])
df = stratified_subsample(df, cfg["data"]["max_runs"], cfg["seed"])
df["split"] = make_splits(df, **cfg["split"], seed=cfg["seed"]).values

out = cfg["paths"]["processed"] / "runs.parquet"
df.to_parquet(out, index=False)

print(f"\nSaved {len(df):,} runs -> {out}")
print(f"instances: {df.instance_id.nunique():,}   models: {df.model_name.nunique()}")
print(f"success rate: {df.target.mean():.3f}   median steps: {df.n_steps.median():.0f}")
print("\nsplit sizes / success rate:")
print(df.groupby("split")["target"].agg(["size", "mean"]).round(3))
sym = df["symbols"].explode().astype(int).value_counts(normalize=True).sort_index()
print("\nsymbol frequencies:")
for i, p in sym.items():
    print(f"  {SYMBOLS[i]:<14} {p:.3f}")
