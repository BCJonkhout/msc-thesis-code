"""Histograms of pairwise Kendall tau-b for NovelQA-only, QASPER-only, cross-dataset.

Three stacked panels of pairwise tau-b on the 36 candidate-pairs per view.
Vertical lines mark the pre-registered band boundaries (0.33, 0.67) and the
observed median for each view.

Reads:  code/outputs/sanity/kendall_cross_dataset_under_gold_20260520.json
Writes: thesis-msc/figures/results/kendall_distribution.{pdf,png}

Provenance (see docs/CODEMAP.md): RETIRED pilot figure. Reads the same pilot-era
cross-dataset Kendall bundle whose headline (a cross-dataset rank "inversion") was
REFUTED by the main study. No current thesis-msc/**/*.tex references this figure;
canonical result figures come from figures_main_study.py. Kept for history only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "code" / "outputs" / "sanity" / "kendall_cross_dataset_under_gold_20260520.json"
OUT_DIR = REPO / "thesis-msc" / "figures" / "results"
OUT_STEM = OUT_DIR / "kendall_distribution"

VIEW_CONFIG = [
    ("novelqa_only", "NovelQA-only (4-arch, n=15)", "#1f77b4"),
    ("qasper_only", "QASPER-only (4-arch, n=20)", "#d62728"),
    (
        "cross_dataset_8cell",
        "Cross-dataset (8-cell, QASPER + NovelQA)",
        "#2ca02c",
    ),
]
LOW_BAND = 0.33
HIGH_BAND = 0.67


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.2), sharex=True)
    bins = np.linspace(-1.05, 1.05, 22)

    for ax, (block_key, label, color) in zip(axes, VIEW_CONFIG):
        block = payload["blocks"][block_key]
        taus = [pair["tau"] for pair in block["pairs"]]
        median = float(np.median(taus))

        ax.hist(
            taus,
            bins=bins,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.axvline(LOW_BAND, color="grey", linestyle="--", linewidth=0.9)
        ax.axvline(HIGH_BAND, color="grey", linestyle="--", linewidth=0.9)
        ax.axvline(median, color="black", linewidth=1.6)
        ax.text(
            LOW_BAND,
            ax.get_ylim()[1] if False else 0.94,
            f" band-low {LOW_BAND}",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
            color="grey",
        )
        ax.text(
            HIGH_BAND,
            0.94,
            f" band-high {HIGH_BAND}",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
            color="grey",
        )
        if median > 0.7:
            ax.text(
                median,
                0.82,
                f"median = {median:.3f} ",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=9,
                color="black",
            )
        else:
            ax.text(
                median,
                0.82,
                f" median = {median:.3f}",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=9,
                color="black",
            )
        ax.set_title(f"{label}  ({len(taus)} pairs, mean = {np.mean(taus):.3f})", fontsize=10)
        ax.set_ylabel("Pair count")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Pairwise Kendall $\\tau_b$ across candidate answerers")
    fig.suptitle(
        "Distribution of pairwise Kendall $\\tau_b$ on the same 9-candidate slate, three views",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_STEM.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {OUT_STEM.with_suffix('.pdf')}")
    print(f"wrote {OUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
