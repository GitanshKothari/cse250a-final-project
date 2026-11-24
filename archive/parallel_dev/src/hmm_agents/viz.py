from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .hmm import CategoricalHMM
from .parsing import SYMBOLS


def plot_hmm(hmm: CategoricalHMM, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    K = hmm.n_states
    states = [f"S{k}" for k in range(K)]

    fig, ax = plt.subplots(figsize=(10, 0.5 * K + 1.5))
    sns.heatmap(pd.DataFrame(hmm.emissionprob_, index=states, columns=SYMBOLS),
                annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Emission probabilities P(symbol | state)")
    fig.tight_layout(); fig.savefig(out_dir / "hmm_emissions.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(0.7 * K + 2, 0.6 * K + 1.5))
    sns.heatmap(pd.DataFrame(hmm.transmat_, index=states, columns=states),
                annot=True, fmt=".2f", cmap="Purples", cbar=False, ax=ax)
    ax.set_xlabel("to"); ax.set_ylabel("from")
    ax.set_title("Transition probabilities")
    fig.tight_layout(); fig.savefig(out_dir / "hmm_transitions.png", dpi=150); plt.close(fig)


def plot_occupancy(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    summary[["success", "failure"]].plot.bar(ax=ax, color=["#2a9d8f", "#c44536"])
    ax.set_ylabel("mean fraction of steps"); ax.set_title("Time spent in each hidden state")
    ax.set_xticklabels([f"{i}\n{s.split(',')[0]}" for i, s in
                        zip(summary.index, summary.top_symbols)], rotation=0, fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "state_occupancy.png", dpi=150); plt.close(fig)


def plot_auroc(results: pd.DataFrame, out_dir: Path, split: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in results.groupby("features"):
        ax.plot(g.t, g.auroc, marker="o", label=name)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("prefix length t (steps observed)"); ax.set_ylabel(f"AUROC ({split})")
    ax.set_title("Early failure prediction"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"auroc_vs_t_{split}.png", dpi=150); plt.close(fig)
