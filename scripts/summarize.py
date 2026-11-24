from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _base(m: pd.Series) -> pd.Series:
    return m.str.rsplit("@s", n=1).str[0]


def summarize(dirs: list[Path], ts: list[int], trade_t: int = 10, stop: float = 0.1) -> pd.DataFrame:
    rows, seen = [], set()
    for d in dirs:
        met = pd.read_csv(d / "metrics.csv")
        met = met[met.cohort == "alive"].assign(item=lambda x: _base(x["method"]))
        tr = pd.read_csv(d / "tradeoff.csv").assign(item=lambda x: _base(x["method"]))
        tr = tr[(tr.t == trade_t) & (tr.stop_frac.round(3) == stop)]
        for item, g in met.groupby("item", sort=False):
            if item in seen:
                continue
            seen.add(item)
            row = {"method": item, "source": d.parent.name if d.name in ("cv", "val") else d.name,
                   "n_seeds": g["method"].nunique()}
            for t in ts:
                gt = g[g.t == t]
                for col, lab in [("auroc", "pooled"), ("auroc_wi", "within")]:
                    if col in gt and len(gt):
                        v = gt[col]
                        row[f"{lab}@{t}"] = v.mean()
                        row[f"{lab}@{t}_range"] = v.max() - v.min()
            gtr = tr[tr.item == item]
            if len(gtr):
                row["succ_lost@10%"] = gtr["successes_lost"].mean()
                row["compute_saved@10%"] = gtr["compute_saved"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, ts: list[int]) -> str:
    cols = ["method", "n_seeds"] + [f"{k}@{t}" for t in ts for k in ("pooled", "within")] + \
        ["succ_lost@10%", "compute_saved@10%"]
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if c in ("method",):
                cells.append(str(v))
            elif c == "n_seeds":
                cells.append(str(int(v)))
            elif pd.isna(v):
                cells.append("")
            else:
                cells.append(f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--t", nargs="+", type=int, default=[5, 10, 20, 30])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    df = summarize(a.dirs, a.t)
    md = to_markdown(df, a.t)
    print(md)
    if a.out:
        a.out.write_text(md + "\n")
        df.to_csv(a.out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
