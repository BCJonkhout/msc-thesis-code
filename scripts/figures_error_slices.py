"""Paper-ready RQ2 error-slice figure: the structure penalty by question
granularity on NovelQA (gist -> fine-grained detail), driven entirely from the
on-disk scored cells and NovelQA's own question labels (no hardcoded numbers).

Outputs into code/outputs/main_study/export/ (promoted to thesis-msc/generated/
by make export-assets):
  mainstudy_error_slices.pdf / .png  -> fig:results-error-slices

The grouping is a conceptual binning of NovelQA's aspect labels by how much
verbatim / exhaustive access the answer needs; membership is fixed below and
documented in the paper. QASPER is intentionally not drawn here: its gaps are
uniformly small (no granularity gradient), reported as an appendix table.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "outputs" / "main_study"
DATA = ROOT / "data"
EXPORT = MS / "export"
EXPORT.mkdir(parents=True, exist_ok=True)

ARCH = ["flat", "naive_rag", "raptor", "graphrag"]
LBL = {"flat": "Flat", "naive_rag": "Naive RAG", "raptor": "RAPTOR", "graphrag": "GraphRAG"}
COL = {"flat": "#0072B2", "naive_rag": "#009E73", "raptor": "#E69F00", "graphrag": "#D55E00"}

# granularity bins (gist -> fine detail), by what each NovelQA aspect demands
BINS = [
    ("Gist & relational\n(plot, relat)", ["plot", "relat"]),
    ("Entity & scene facts\n(character, setting)", ["character", "settg"]),
    ("Fine-grained detail\n(meaning, counting, span)", ["meaning", "times", "span"]),
]


def load():
    m = defaultdict(dict)
    for line in (MS / "scored_cells.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        m[(d["dataset"], d["cluster"], d["qid"])][d["arch"]] = d["metric"]
    asp = {}
    for line in (DATA / "novelqa" / "questions.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        asp[(d["novel_id"], d["question_id"])] = d.get("Aspect")
    bins = []
    for label, members in BINS:
        acc = defaultdict(list)
        for k, v in m.items():
            if k[0] == "novelqa" and all(a in v for a in ARCH) and asp.get((k[1], k[2])) in members:
                for a in ARCH:
                    acc[a].append(v[a])
        n = len(acc["flat"])
        bins.append((label, {a: sum(acc[a]) / n for a in ARCH}, n))
    return bins


def main() -> None:
    bins = load()
    x = np.arange(len(bins))
    flat_y = [b[1]["flat"] for b in bins]
    worst_y = [min(b[1][a] for a in ["naive_rag", "raptor", "graphrag"]) for b in bins]

    fig, ax = plt.subplots(figsize=(9, 6))
    for a in ARCH:
        ax.plot(x, [b[1][a] for b in bins], "-o", color=COL[a], lw=2.4, ms=9,
                label=LBL[a], zorder=3, markeredgecolor="white", markeredgewidth=0.8)

    def whisker(xc, lo, hi, capw=0.035, color="#999"):
        ax.plot([xc, xc], [lo, hi], color=color, lw=1.6, zorder=5)
        for yy in (lo, hi):
            ax.plot([xc - capw, xc + capw], [yy, yy], color=color, lw=1.6, zorder=5)

    for xi, (fl, wo) in enumerate(zip(flat_y, worst_y)):
        xw = xi - 0.10 if xi == 0 else xi + 0.10
        whisker(xw, wo, fl)
        if xi == 1:
            ax.text(xi + 0.18, fl, f"gap {fl-wo:.2f}", ha="left", va="top",
                    fontsize=10, fontweight="bold", color="#444")
        elif xi == 0:
            ax.text(xw - 0.07, (fl + wo) / 2, f"gap {fl-wo:.2f}", ha="right", va="center",
                    fontsize=10, fontweight="bold", color="#444")
        else:
            ax.text(xw + 0.07, (fl + wo) / 2, f"gap {fl-wo:.2f}", ha="left", va="center",
                    fontsize=10, fontweight="bold", color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b[0]}\n(n={b[2]})" for b in bins], fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(-0.55, len(bins) - 0.30)
    # No baked title: the LaTeX figure caption carries the message in the paper.
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=1)
    ax.grid(axis="y", color="#eee")
    ax.annotate("", xy=(len(bins) - 0.85, 0.355), xytext=(-0.15, 0.355),
                arrowprops=dict(arrowstyle="->", color="#999"))
    ax.text((len(bins) - 1) / 2, 0.33, "more fine-grained / verbatim demand →",
            ha="center", fontsize=8, color="#777")

    fig.tight_layout()
    fig.savefig(EXPORT / "mainstudy_error_slices.pdf")
    fig.savefig(EXPORT / "mainstudy_error_slices.png", dpi=150)
    print("bins:", [(b[0].split(chr(10))[0], round(b[1]["flat"], 2),
                     round(min(b[1][a] for a in ["naive_rag", "raptor", "graphrag"]), 2), b[2]) for b in bins])
    print("wrote mainstudy_error_slices.{pdf,png} to", EXPORT)


if __name__ == "__main__":
    main()
