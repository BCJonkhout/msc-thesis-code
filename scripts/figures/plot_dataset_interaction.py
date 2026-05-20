"""Scatter of per-(candidate, architecture) QASPER vs NovelQA accuracy.

Each point is one (candidate, architecture) cell. Colour encodes architecture.
The y=x diagonal would mean "same cell wins on both datasets" -- large
off-diagonal scatter visualises the architecture-by-workload interaction.
QASPER is plotted in its native Answer-F1 range, NovelQA in its native
multiple-choice accuracy range; per-architecture centroids are overlaid.

Reads:  code/outputs/sanity/kendall_cross_dataset_under_gold_20260520.json
Writes: thesis-msc/figures/results/dataset_interaction.{pdf,png}
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "code" / "outputs" / "sanity" / "kendall_cross_dataset_under_gold_20260520.json"
OUT_DIR = REPO / "thesis-msc" / "figures" / "results"
OUT_STEM = OUT_DIR / "dataset_interaction"

ARCHS = ["flat", "naive_rag", "raptor", "graphrag"]
ARCH_LABELS = {
    "flat": "Flat",
    "naive_rag": "Naive RAG",
    "raptor": "RAPTOR",
    "graphrag": "GraphRAG",
}
ARCH_COLORS = {
    "flat": "#1f77b4",
    "naive_rag": "#d62728",
    "raptor": "#2ca02c",
    "graphrag": "#9467bd",
}


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    qasper = payload["blocks"]["qasper_only"]["score_table"]
    novelqa = payload["blocks"]["novelqa_only"]["score_table"]
    candidates = payload["candidates_common"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))

    # diagonal reference (rescaled axes use the visual extent of the plot)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0, alpha=0.7, label="y = x")

    for arch in ARCHS:
        xs = [qasper[c][arch] for c in candidates]
        ys = [novelqa[c][arch] for c in candidates]
        ax.scatter(
            xs,
            ys,
            s=50,
            color=ARCH_COLORS[arch],
            edgecolors="black",
            linewidths=0.5,
            alpha=0.85,
            label=ARCH_LABELS[arch],
            zorder=3,
        )
        # centroid
        cx = statistics.fmean(xs)
        cy = statistics.fmean(ys)
        ax.scatter(
            [cx],
            [cy],
            s=180,
            marker="X",
            color=ARCH_COLORS[arch],
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )
        ax.annotate(
            ARCH_LABELS[arch],
            xy=(cx, cy),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=ARCH_COLORS[arch],
            fontweight="bold",
        )

    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("QASPER Answer-F1 (short docs, ~10k tokens)")
    ax.set_ylabel("NovelQA accuracy (long docs, ~250k tokens)")
    ax.set_title(
        "Per-(candidate, architecture) cells: dataset-conditional architecture ranking",
        fontsize=10.5,
    )
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT_STEM.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {OUT_STEM.with_suffix('.pdf')}")
    print(f"wrote {OUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
