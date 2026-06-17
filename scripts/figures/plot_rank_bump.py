"""Rank-bump chart: per-candidate architecture rank, two panels (QASPER, NovelQA).

x-axis = candidate (grouped by vendor); y-axis = architecture rank within
dataset (1 = best, 4 = worst). One coloured line per architecture; two side-by-
side panels visualise within-dataset stability versus the cross-dataset
inversion.

Reads:  code/outputs/sanity/kendall_cross_dataset_under_gold_20260520.json
Writes: thesis-msc/figures/results/rank_bump.{pdf,png}

Provenance (see docs/CODEMAP.md): RETIRED pilot figure. The cross-dataset
architecture-rank inversion this chart visualises was REFUTED by the main study
(flat wins both QASPER and NovelQA; RAPTOR/GraphRAG dominated). No current
thesis-msc/**/*.tex references it; canonical figures come from
figures_main_study.py. Kept for history only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "code" / "outputs" / "sanity" / "kendall_cross_dataset_under_gold_20260520.json"
OUT_DIR = REPO / "thesis-msc" / "figures" / "results"
OUT_STEM = OUT_DIR / "rank_bump"

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

# vendor ordering puts each family contiguous
VENDOR_ORDER = [
    ("DeepSeek", ["deepseek-v4-flash", "deepseek-v4-pro"]),
    ("Google", ["gemini-3.1-flash-lite-preview", "gemini-flash-latest"]),
    (
        "xAI",
        [
            "grok-4-1-fast-non-reasoning",
            "grok-4-fast-reasoning",
            "grok-4.20-0309-non-reasoning",
            "grok-4.20-0309-reasoning",
            "grok-4.3",
        ],
    ),
]

SHORT_LABEL = {
    "deepseek-v4-flash": "ds-flash",
    "deepseek-v4-pro": "ds-pro",
    "gemini-3.1-flash-lite-preview": "g-3.1-fl",
    "gemini-flash-latest": "g-fl-lt",
    "grok-4-1-fast-non-reasoning": "grok-4.1-nr",
    "grok-4-fast-reasoning": "grok-4-r",
    "grok-4.20-0309-non-reasoning": "grok-4.20-nr",
    "grok-4.20-0309-reasoning": "grok-4.20-r",
    "grok-4.3": "grok-4.3",
}


def candidate_order() -> list[str]:
    out: list[str] = []
    for _, cands in VENDOR_ORDER:
        out.extend(cands)
    return out


def per_arch_ranks(score_table: dict, candidates: list[str]) -> dict[str, list[int]]:
    """Return arch -> list of ranks across candidates (1=best)."""
    out: dict[str, list[int]] = {arch: [] for arch in ARCHS}
    for cand in candidates:
        cell = score_table[cand]
        # descending sort -> rank 1 is largest score
        ordered = sorted(ARCHS, key=lambda a: -cell[a])
        rank = {arch: i + 1 for i, arch in enumerate(ordered)}
        for arch in ARCHS:
            out[arch].append(rank[arch])
    return out


def draw_panel(ax, score_table: dict, candidates: list[str], title: str) -> None:
    ranks = per_arch_ranks(score_table, candidates)
    x = np.arange(len(candidates))
    for arch in ARCHS:
        ax.plot(
            x,
            ranks[arch],
            marker="o",
            markersize=6,
            linewidth=1.8,
            color=ARCH_COLORS[arch],
            label=ARCH_LABELS[arch],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABEL[c] for c in candidates], rotation=40, ha="right")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(4.6, 0.2)  # inverted: rank 1 at top, extra headroom for vendor labels
    ax.set_ylabel("Architecture rank within dataset (1 = best)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    # vendor separators
    cursor = 0
    for vendor, cands in VENDOR_ORDER:
        cursor += len(cands)
        if cursor < len(candidates):
            ax.axvline(cursor - 0.5, color="grey", alpha=0.4, linewidth=0.7, linestyle="--")
    # vendor labels along top
    cursor = 0
    for vendor, cands in VENDOR_ORDER:
        mid = cursor + (len(cands) - 1) / 2
        ax.text(
            mid,
            0.45,
            vendor,
            ha="center",
            va="top",
            fontsize=8.5,
            color="dimgrey",
            fontweight="bold",
            transform=ax.transData,
        )
        cursor += len(cands)


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    qasper = payload["blocks"]["qasper_only"]["score_table"]
    novelqa = payload["blocks"]["novelqa_only"]["score_table"]
    candidates = candidate_order()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    draw_panel(axes[0], qasper, candidates, "QASPER (~10k-token research papers)")
    draw_panel(axes[1], novelqa, candidates, "NovelQA (~250k-token novels)")
    axes[1].set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Within-dataset architecture-rank stability across candidates; ranking inverts between datasets",
        y=1.08,
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(OUT_STEM.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {OUT_STEM.with_suffix('.pdf')}")
    print(f"wrote {OUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
