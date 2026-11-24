import argparse

from hmm_agents.config import ROOT, load_config
from hmm_agents.data import N_SHARDS, build_runs_table, stratified_subsample
from hmm_agents.parsing import SYMBOLS
from hmm_agents.splits import assign_splits, load_split_table, split_fractions

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
sc = cfg["split"]
split_file = ROOT / sc["file"]
if not split_file.exists():
    raise SystemExit(f"{split_file} missing. Run scripts/00_make_splits.py once (it is committed).")
fractions = split_fractions(sc["test"], sc["val"], sc["hmm_frac_of_train"])
split, n_fallback = assign_splits(df["instance_id"], load_split_table(split_file), fractions)
df["split"] = split.values
if n_fallback:
    print(f"WARNING: {n_fallback} issues not in {split_file.name}; hash-assigned (stable).")

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
