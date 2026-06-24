"""Paper-ready figures for the main study, all driven from the on-disk analysis
JSONs (no hardcoded metrics): significance.json (quality + clustered-bootstrap
CIs), cost_per_arch.json (deployment cost), breakeven.json (amortised curves).

Emits vector PDF (for \\input into the paper via generated/) plus PNG previews
into code/outputs/main_study/figures/:
  - pareto_cost_quality.pdf : deployment cost vs quality, both datasets, frontier
  - accuracy_by_arch.pdf    : per-arch quality with 95% clustered-bootstrap CIs
  - breakeven_curves.pdf    : amortised cost/query vs questions-per-document
  - cost_composition.pdf    : where cost comes from -- build + per-query, by source

A consistent architecture colour/marker scheme is shared across all three so the
figures read as a set. Frontier architectures (flat, naive_rag) get filled
markers; dominated ones (raptor, graphrag) get hollow markers + an x.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "outputs" / "main_study"
FIG = MS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ARCHS = ["flat", "naive_rag", "raptor", "graphrag"]
LABEL = {"flat": "Flat", "naive_rag": "Naive RAG", "raptor": "RAPTOR", "graphrag": "GraphRAG"}
# colourblind-safe (Wong); consistent across every figure
COLOR = {"flat": "#0072B2", "naive_rag": "#009E73", "raptor": "#E69F00", "graphrag": "#D55E00"}
FRONTIER = {"flat", "naive_rag"}          # Pareto-optimal
MARKER = {"flat": "o", "naive_rag": "s", "raptor": "^", "graphrag": "D"}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

sig = json.loads((MS / "significance.json").read_text(encoding="utf-8"))
cost = json.loads((MS / "cost_per_arch.json").read_text(encoding="utf-8"))
brk = json.loads((MS / "breakeven.json").read_text(encoding="utf-8"))
mem = json.loads((MS / "memorization_control.json").read_text(encoding="utf-8"))

QUAL = {  # dataset -> arch -> (mean, lo, hi)
    ds: {a: (d["per_arch"][a]["mean"], d["per_arch"][a]["ci_low"], d["per_arch"][a]["ci_high"])
         for a in ARCHS}
    for ds, d in sig["datasets"].items()
}
TOTAL = {a: cost["per_arch"][f"base|{a}"]["total"] for a in ARCHS}          # deployment USD
FLOOR = {ds: mem[ds]["closed_book"] for ds in ("qasper", "novelqa")}        # no-document floor
DS_TITLE = {"qasper": "QASPER (Answer-F1)", "novelqa": "NovelQA (accuracy)"}
DS_SHORT = {"qasper": "QASPER", "novelqa": "NovelQA"}


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def _marker_kw(a: str) -> dict:
    """Filled for frontier archs, hollow for dominated ones."""
    if a in FRONTIER:
        return dict(marker=MARKER[a], markerfacecolor=COLOR[a], markeredgecolor=COLOR[a])
    return dict(marker=MARKER[a], markerfacecolor="white", markeredgecolor=COLOR[a], markeredgewidth=1.6)


def fig_pareto() -> None:
    """Cost (log x) vs quality (y), one panel per dataset; frontier highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, ds in zip(axes, ("qasper", "novelqa")):
        # frontier line: the non-dominated set, sorted by cost (naive_rag -> flat)
        front = sorted(FRONTIER, key=lambda a: TOTAL[a])
        ax.plot([TOTAL[a] for a in front], [QUAL[ds][a][0] for a in front],
                "-", color="0.55", lw=1.3, zorder=1, label="Pareto frontier")
        for a in ARCHS:
            m, lo, hi = QUAL[ds][a]
            ax.errorbar(TOTAL[a], m, yerr=[[m - lo], [hi - m]], elinewidth=1.1,
                        capsize=2.5, ecolor=COLOR[a], color=COLOR[a], zorder=2,
                        linestyle="none", markersize=8, **_marker_kw(a))
            dom = "" if a in FRONTIER else "  (dominated)"
            va, dy = ("bottom", 1.012) if a != "raptor" else ("top", 0.988)
            ax.annotate(f"{LABEL[a]}{dom}", (TOTAL[a], m * dy),
                        fontsize=8.2, ha="center", va=va, color=COLOR[a])
        ax.set_xscale("log")
        ax.set_xlabel("Deployment cost (USD, log scale)")
        ax.set_ylabel(DS_TITLE[ds])
        ax.set_title(DS_SHORT[ds], fontsize=10)
        ax.margins(y=0.18)
    handles = [Line2D([0], [0], color="0.55", lw=1.3, label="Pareto frontier"),
               Line2D([0], [0], marker="o", color="0.3", linestyle="none",
                      markerfacecolor="0.3", label="on frontier"),
               Line2D([0], [0], marker="^", color="0.3", linestyle="none",
                      markerfacecolor="white", markeredgecolor="0.3", label="dominated")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Cost–quality Pareto: frontier = {Flat, Naive RAG}; RAPTOR and GraphRAG dominated",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "pareto_cost_quality")


def fig_accuracy() -> None:
    """Grouped bars per dataset with 95% clustered-bootstrap CI whiskers."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, ds in zip(axes, ("qasper", "novelqa")):
        for i, a in enumerate(ARCHS):
            m, lo, hi = QUAL[ds][a]
            ax.bar(i, m, color=COLOR[a], width=0.66, alpha=0.92,
                   yerr=[[m - lo], [hi - m]], capsize=4,
                   error_kw=dict(elinewidth=1.2, ecolor="0.2"))
            ax.text(i, hi + (0.012 if ds == "qasper" else 0.018), f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8.4)
        ax.set_xticks(range(len(ARCHS)))
        ax.set_xticklabels([LABEL[a] for a in ARCHS], fontsize=9)
        ax.set_ylabel(DS_TITLE[ds])
        ax.set_title(DS_SHORT[ds], fontsize=10)
        top = max(QUAL[ds][a][2] for a in ARCHS)
        ax.set_ylim(0, top * 1.16)
    fig.suptitle("Per-architecture quality with 95% clustered-bootstrap CIs (identical ranking on both datasets)",
                 fontsize=10.5, y=1.01)
    fig.tight_layout()
    _save(fig, "accuracy_by_arch")


def fig_breakeven() -> None:
    """Amortised deployment cost per query vs questions-per-document; both price cards."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    flat_q = cost["per_arch"]["base|flat"]["c_on_per_query"]
    flat_q_cache = cost["per_arch"]["cache|flat"]["c_on_per_query"]
    Ns = [n / 2 for n in range(2, 61)]  # 1.0 .. 30.0
    # Flat's per-query cost under each price card (the break-even targets).
    ax.axhline(flat_q, color=COLOR["flat"], lw=2.0,
               label=f"Flat, standard card (${flat_q*1000:.2f}m/q)")
    ax.axhline(flat_q_cache, color=COLOR["flat"], lw=1.6, ls="--",
               label=f"Flat, cache card (${flat_q_cache*1000:.2f}m/q)")
    for a in ("naive_rag", "raptor", "graphrag"):
        b = brk[f"base|{a}"]
        coff_doc, onq, nstar = b["c_off_per_doc"], b["c_on_per_query"], b["n_star"]
        ys = [coff_doc / n + onq for n in Ns]
        ax.plot(Ns, ys, color=COLOR[a], lw=2.0, label=LABEL[a])
        if nstar and 1 <= nstar <= 30:  # standard-card crossing
            yat = coff_doc / nstar + onq
            ax.plot([nstar], [yat], "o", color=COLOR[a], markersize=7, zorder=5)
            ax.annotate(f"$N^*\\approx{nstar:.1f}$", (nstar, yat), fontsize=8.6,
                        color=COLOR[a], xytext=(6, 6), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlim(1, 30)
    ymin, ymax = ax.get_ylim()
    # real operating densities (labels at the top; the legend now sits outside the axes)
    for x, txt in ((4, "QASPER $\\approx$ 4 q/paper"), (25, "NovelQA $\\approx$ 25 q/novel")):
        ax.axvline(x, color="0.6", ls=":", lw=1.0)
        ax.text(x - 0.4, ymax * 0.9, txt, rotation=90, va="top", ha="right",
                fontsize=7.6, color="0.4")
    ax.set_xlabel("Questions per document $N$ (build cost amortised over $N$ queries)")
    ax.set_ylabel("Amortised deployment cost per query (USD, log scale)")
    ax.set_title("Break-even vs Flat: structured builds pay back only at high $N$", fontsize=10.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.4, borderaxespad=0.0)
    fig.tight_layout()
    _save(fig, "breakeven_curves")


def fig_memorization() -> None:
    """Per-architecture quality with the closed-book (no-document) floor drawn in.

    Bars from zero (honest magnitude, no truncation) with 95% clustered-bootstrap
    CIs; the floor is a dashed line across each panel and the band below it is
    shaded as the recall region. This figure exists to show how much of each score
    is just the floor, so the axis is not zoomed: on NovelQA most of the bar lies
    below the floor (memorised recall) and only Flat rises clearly above it, while
    on QASPER the floor is low so the bars are chiefly reading-derived. The shared
    zero baseline keeps the two panels visually comparable."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, ds in zip(axes, ("qasper", "novelqa")):
        floor = FLOOR[ds]
        top = max(QUAL[ds][a][2] for a in ARCHS)
        ax.set_ylim(0, top * 1.16)
        ax.axhspan(0, floor, color="0.5", alpha=0.10, zorder=0)  # recall region
        for i, a in enumerate(ARCHS):
            m, lo, hi = QUAL[ds][a]
            ax.bar(i, m, color=COLOR[a], width=0.66, alpha=0.92, zorder=2,
                   yerr=[[m - lo], [hi - m]], capsize=4,
                   error_kw=dict(elinewidth=1.2, ecolor="0.2"))
            ax.text(i, hi + (0.012 if ds == "qasper" else 0.018), f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8.4)
        ax.axhline(floor, color="0.2", ls="--", lw=1.4, zorder=3)
        ax.annotate(f"closed-book floor = {floor:.2f}", (3.45, floor),
                    fontsize=8.0, ha="right",
                    va="bottom" if ds == "qasper" else "top", color="0.2")
        ax.set_xticks(range(len(ARCHS)))
        ax.set_xticklabels([LABEL[a] for a in ARCHS], fontsize=9)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylabel(DS_TITLE[ds])
        ax.set_title(DS_SHORT[ds], fontsize=10)
    fig.suptitle("Answer quality against the closed-book floor (bars from zero; shaded band = memorised-recall floor)",
                 fontsize=9.8, y=1.02)
    fig.tight_layout()
    _save(fig, "memorization_floor")


def fig_cost_composition() -> None:
    """Two stacked-bar panels showing where each architecture's cost comes from.

    (a) one-time build cost (USD) by source -- LLM extraction/summarization vs index
    embedding; (b) per-query answering cost (milli-USD) by ledger source -- uncached
    input, cached input, output, retrieval embedding. The cached-input segment is
    visible only for Flat: the quantitative form of which ledger counter each method
    lights up. Standard price card, post-exclusion evaluation pool."""
    PCB = cost["per_arch"]
    NQ = cost["n_questions"]
    EMB = "#009E73"
    fig, (axB, axQ) = plt.subplots(1, 2, figsize=(9.8, 4.3))
    x = list(range(len(ARCHS)))
    xl = [LABEL[a] for a in ARCHS]

    # (a) one-time build cost, USD, stacked LLM + embedding
    build_src = [("LLM (extraction / summarization)", "gemini_off", "#7B3294"),
                 ("embedding (index / tree)", "embed_off", EMB)]
    bottom = [0.0] * len(ARCHS)
    for name, key, col in build_src:
        vals = [PCB[f"base|{a}"][key] for a in ARCHS]
        axB.bar(x, vals, bottom=bottom, width=0.62, color=col, label=name,
                edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, a in enumerate(ARCHS):
        tot = PCB[f"base|{a}"]["c_off_total"]
        axB.text(i, tot, f"${tot:.2f}", ha="center", va="bottom", fontsize=8.2)
    axB.set_xticks(x); axB.set_xticklabels(xl, fontsize=9)
    axB.set_ylabel(r"One-time build cost $C_{\mathrm{off}}$ (USD)")
    axB.set_title("(a) Build: one-time dollars, almost all LLM calls", fontsize=9.4, pad=8)
    axB.legend(fontsize=7.4, loc="upper left")
    axB.set_ylim(0, max(PCB[f"base|{a}"]["c_off_total"] for a in ARCHS) * 1.22)

    # (b) per-query answering cost, milli-USD, stacked by ledger source
    perq_src = [("uncached input (cold read)", "c_on_uncached", "#0072B2"),
                ("cached input (warm re-read)", "c_on_cached", "#E69F00"),
                ("output", "c_on_output", "#D55E00"),
                ("embedding (retrieval)", "embed_on", EMB)]
    bottom = [0.0] * len(ARCHS)
    for name, key, col in perq_src:
        vals = [PCB[f"base|{a}"][key] / NQ * 1000 for a in ARCHS]
        axQ.bar(x, vals, bottom=bottom, width=0.62, color=col, label=name,
                edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, a in enumerate(ARCHS):
        tot = PCB[f"base|{a}"]["c_on_per_query"] * 1000
        axQ.text(i, tot, f"{tot:.2f}", ha="center", va="bottom", fontsize=8.2)
    axQ.set_xticks(x); axQ.set_xticklabels(xl, fontsize=9)
    axQ.set_ylabel(r"Per-query answering cost $C_{\mathrm{on}}$ (milli-USD)")
    axQ.set_title("(b) Per query: only Flat lights the cached counter", fontsize=9.4, pad=8)
    axQ.legend(fontsize=7.4, loc="upper right")
    axQ.set_ylim(0, max(PCB[f"base|{a}"]["c_on_per_query"] for a in ARCHS) * 1000 * 1.27)

    fig.suptitle("Where each architecture's cost comes from (standard card): "
                 "structured methods pay to build, Flat pays to re-read",
                 fontsize=10.3, y=1.02)
    fig.tight_layout()
    _save(fig, "cost_composition")


def main() -> int:
    print(f"figures -> {FIG}")
    fig_pareto()
    fig_accuracy()
    fig_breakeven()
    fig_memorization()
    fig_cost_composition()
    # tiny provenance stamp so the paper can cite the source JSONs
    (FIG / "SOURCE.txt").write_text(
        "Generated by scripts/figures_main_study.py from outputs/main_study/"
        "{significance,cost_per_arch,breakeven,memorization_control}.json\n", encoding="utf-8")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
