import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from hmm_agents.config import load_config
from hmm_agents.parsing import extract_command, parse_trajectory

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="docs/parser_validation.md")
args = ap.parse_args()
cfg = load_config(args.config)

files = sorted((cfg["paths"]["raw"] / "data").glob("*.parquet"))
if not files:
    raise SystemExit("No raw shards found. Run scripts/01_download.py first.")
table = pq.read_table(files[0], columns=["instance_id", "model_name", "target", "trajectory"])
rng = np.random.default_rng(args.seed)
idx = rng.choice(table.num_rows, size=args.n, replace=False)


def cell(s: str, n: int) -> str:
    return " ".join((s or "").split())[:n].replace("|", "\\|")


lines = ["# Parser validation", "",
         f"{args.n} random runs from `{files[0].name}` (seed {args.seed}). "
         "Mark the `ok?` column with x if the parsed action/error is wrong, and note why.", ""]
for i in idx:
    r = table.slice(int(i), 1).to_pylist()[0]
    steps, _ = parse_trajectory(r["trajectory"], r["instance_id"])
    msgs = [m for m in r["trajectory"] if m["role"] in ("user", "ai")]
    ai = [(j, m) for j, m in enumerate(msgs) if m["role"] == "ai"]
    lines += [f"## {r['instance_id']} | {r['model_name']} | solved={r['target']}", "",
              "| # | command | observation (start) | action | err | ok? |",
              "|---|---|---|---|---|---|"]
    for t, ((j, m), s) in enumerate(zip(ai, steps)):
        obs = msgs[j + 1]["text"] if j + 1 < len(msgs) else ""
        lines.append(f"| {t} | `{cell(extract_command(m['text']), 70)}` | {cell(obs, 90)} "
                     f"| {s.action} | {'E' if s.error else ''} | |")
    lines.append("")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {args.out}")
