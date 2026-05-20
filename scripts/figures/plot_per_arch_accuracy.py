"""Grouped bar chart: per-architecture mean accuracy on QASPER vs NovelQA.

Loads the cross-dataset Kendall input bundle, computes per-architecture
across-candidate means and standard deviations on each dataset, and emits a
grouped bar chart that makes the architecture-rank inversion visible at a
glance.

Reads:  code/outputs/sanity/kendall_cross_dataset_under_gold_20260520.json
Writes: thesis-msc/figures/results/per_arch_accuracy.{pdf,png}
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "code" / "outputs" / "sanity" / "kendall_cross_dataset_under_gold_20260520.json"
OUT_DIR = REPO / "thesis-msc" / "figures" / "results"
OUT_STEM = OUT_DIR / "per_arch_accuracy"

ARCHS = ["flat", "naive_rag", "raptor", "graphrag"]
ARCH_LABELS = {
    "flat": "Flat",
    "naive_rag": "Naive RAG",
    "raptor": "RAPTOR",
    "graphrag": "GraphRAG",
}
DATASETS = [
    ("qasper", "QASPER (Answer-F1)"),
    ("novelqa", "NovelQA (accuracy)"),
]


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    qasper = payload["blocks"]["qasper_only"]["score_table"]
    novelqa = payload["blocks"]["novelqa_only"]["score_table"]

    means: dict[str, list[float]] = {arch: [] for arch in ARCHS}
    sds: dict[str, list[float]] = {arch: [] for arch in ARCHS}

    for dataset_key, _ in DATASETS:
        table = qasper if dataset_key == "qasper" else novelqa
        for arch in ARCHS:
            values = [table[cand][arch] for cand in table]
            means[arch].append(statistics.fmean(values))
            sds[arch].append(statistics.pstdev(values) if len(values) > 1 else 0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))

    n_datasets = len(DATASETS)
    n_archs = len(ARCHS)
    group_width = 0.82
    bar_width = group_width / n_archs
    x = np.arange(n_datasets)

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for i, arch in enumerate(ARCHS):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        bars = ax.bar(
            offsets,
            means[arch],
            width=bar_width,
            yerr=sds[arch],
            capsize=3,
            label=ARCH_LABELS[arch],
            color=palette[i],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, value in zip(bars, means[arch]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS])
    ax.set_ylabel("Mean score across 9 candidate answerers")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        "Per-architecture answer quality on QASPER (short docs) vs. NovelQA (long docs)",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_STEM.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {OUT_STEM.with_suffix('.pdf')}")
    print(f"wrote {OUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
