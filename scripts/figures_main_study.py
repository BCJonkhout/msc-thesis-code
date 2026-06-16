"""Paper-ready figures for the main study, all driven from the on-disk analysis
JSONs (no hardcoded metrics): significance.json (quality + clustered-bootstrap
CIs), cost_per_arch.json (deployment cost), breakeven.json (amortised curves).

Emits vector PDF (for \\input into the paper via generated/) plus PNG previews
into code/outputs/main_study/figures/:
  - pareto_cost_quality.pdf : deployment cost vs quality, both datasets, frontier
  - accuracy_by_arch.pdf    : per-arch quality with 95% clustered-bootstrap CIs
  - breakeven_curves.pdf    : amortised cost/query vs questions-per-document

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

QUAL = {  # dataset -> arch -> (mean, lo, hi)
    ds: {a: (d["per_arch"][a]["mean"], d["per_arch"][a]["ci_low"], d["per_arch"][a]["ci_high"])
         for a in ARCHS}
    for ds, d in sig["datasets"].items()
}
TOTAL = {a: cost["per_arch"][f"base|{a}"]["total"] for a in ARCHS}          # deployment USD
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
    """Amortised deployment cost per query vs questions-per-document (base card)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    flat_q = cost["per_arch"]["base|flat"]["c_on_per_query"]
    Ns = [n / 2 for n in range(2, 61)]  # 1.0 .. 30.0
    ax.axhline(flat_q, color=COLOR["flat"], lw=2.0, label=f"Flat (no build): ${flat_q*1000:.2f}m/q")
    for a in ("naive_rag", "raptor", "graphrag"):
        b = brk[f"base|{a}"]
        coff_doc, onq, nstar = b["c_off_per_doc"], b["c_on_per_query"], b["n_star"]
        ys = [coff_doc / n + onq for n in Ns]
        ax.plot(Ns, ys, color=COLOR[a], lw=2.0, label=LABEL[a])
        if nstar and 1 <= nstar <= 30:
            yat = coff_doc / nstar + onq
            ax.plot([nstar], [yat], "o", color=COLOR[a], markersize=7, zorder=5)
            ax.annotate(f"$N^*\\approx{nstar:.1f}$", (nstar, yat), fontsize=8.6,
                        color=COLOR[a], xytext=(6, 6), textcoords="offset points")
    # real operating densities
    for x, txt in ((4, "QASPER ≈ 4 q/paper"), (25, "NovelQA ≈ 25 q/novel")):
        ax.axvline(x, color="0.6", ls=":", lw=1.0)
        ax.text(x, ax.get_ylim()[1], txt, rotation=90, va="top", ha="right",
                fontsize=7.6, color="0.4")
    ax.set_yscale("log")
    ax.set_xlabel("Questions per document $N$ (build cost amortised over $N$ queries)")
    ax.set_ylabel("Amortised deployment cost per query (USD, log scale)")
    ax.set_xlim(1, 30)
    ax.set_title("Break-even vs Flat: structure pays only above $N^*$ queries per document", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.8)
    fig.tight_layout()
    _save(fig, "breakeven_curves")


def main() -> int:
    print(f"figures -> {FIG}")
    fig_pareto()
    fig_accuracy()
    fig_breakeven()
    # tiny provenance stamp so the paper can cite the source JSONs
    (FIG / "SOURCE.txt").write_text(
        "Generated by scripts/figures_main_study.py from outputs/main_study/"
        "{significance,cost_per_arch,breakeven}.json\n", encoding="utf-8")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
