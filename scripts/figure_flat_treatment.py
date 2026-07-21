"""Companion figure to breakeven_flat_build_treatment.py: what the break-even
panels look like when Flat is drawn under each cost treatment.

Per dataset (base card):
  - Flat, realized-mean treatment: horizontal line at the realized per-query
    mean (the paper's treatment).
  - Flat, first-read-as-build treatment: its own amortizing curve
    c(n) = c_on_alt + C_off^flat/n. The two coincide exactly at the observed
    question density (marked), and only there.
  - Ideal-cache envelope: the same curve with every re-read served from cache
    (the alternative treatment's premise); the gap to the dashed curve is the
    realized cache-miss cost, which recurs per query.
  - Structured-architecture curves with their n* against each Flat reference.

Reads only published artifacts; writes figures/breakeven_flat_treatment.{png,pdf}.
Defense/analysis material -- not exported to the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "outputs" / "main_study"
FIG = MS / "figures"

LABEL = {"naive_rag": "Naive RAG", "raptor": "RAPTOR", "graphrag": "GraphRAG"}
COLOR = {"flat": "#0072B2", "naive_rag": "#009E73", "raptor": "#E69F00", "graphrag": "#D55E00"}

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})

P_UNCACHED, P_CACHED = 0.10e-6, 0.025e-6  # gemini-3.1-flash-lite-preview, base card

alt = json.loads((MS / "breakeven_flat_build_treatment.json").read_text(encoding="utf-8"))
brk = json.loads((MS / "breakeven_by_dataset.json").read_text(encoding="utf-8"))


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    panels = [("qasper", "QASPER", 15, 3.84, 956), ("novelqa", "NovelQA", 80, 25.11, 1381)]
    for ax, (ds, name, xmax, dens, nq) in zip(axes, panels):
        f = alt[f"base|flat|{ds}"]
        flat_mean = f["c_on_per_query_realized"]
        coff_f, on_alt = f["c_off_per_doc"], f["c_on_per_query_alt"]
        # ideal-cache asymptote: subtract the realized re-read cache-miss premium
        miss_usd = f["cache_structure"]["rereads_uncached_tokens"] * (P_UNCACHED - P_CACHED)
        on_ideal = on_alt - miss_usd / nq
        Ns = [1 + (xmax - 1) * i / 320 for i in range(321)]

        ax.axhline(flat_mean, color=COLOR["flat"], lw=2.0,
                   label=f"Flat, realized mean (${flat_mean*1000:.2f}m/q)")
        ax.plot(Ns, [on_alt + coff_f / n for n in Ns], color=COLOR["flat"], lw=1.8, ls="--",
                label="Flat, first read as build")
        ax.plot(Ns, [on_ideal + coff_f / n for n in Ns], color=COLOR["flat"], lw=1.1, ls=":",
                label="Flat, ideal cache (every re-read hits)")
        ax.plot([dens], [flat_mean], "o", color=COLOR["flat"], ms=6.5, zorder=6,
                markerfacecolor="white", markeredgewidth=1.6)

        for a in ("naive_rag", "raptor", "graphrag"):
            b = brk[f"base|{a}|{ds}"]
            coff, onq = b["c_off_per_doc"], b["c_on_per_query"]
            ax.plot(Ns, [coff / n + onq for n in Ns], color=COLOR[a], lw=1.6, label=LABEL[a])
            for nstar, filled in ((b["n_star"], False), (alt[f"base|{a}|{ds}"]["n_star_alt"], True)):
                if nstar and 1 <= nstar <= xmax:
                    y = coff / nstar + onq
                    ax.plot([nstar], [y], "o", color=COLOR[a], ms=5.5, zorder=5,
                            markerfacecolor=COLOR[a] if filled else "white", markeredgewidth=1.4)
                    ax.annotate(f"$n^\\star\\approx{nstar:.0f}$", (nstar, y), fontsize=7.6,
                                color=COLOR[a], xytext=(4, 5 if filled else -11),
                                textcoords="offset points")

        ax.axvline(dens, color="0.55", ls=":", lw=1.1)
        ax.set_yscale("log")
        ax.set_xlim(1, xmax)
        _, ymax = ax.get_ylim()
        ax.text(dens, ymax * 0.85, f"  observed $\\approx${dens:.1f} questions/doc",
                rotation=90, va="top", ha="left", fontsize=7.4, color="0.35")
        ax.set_xlabel("Questions per document $n$")
        ax.set_title(name, fontsize=10.5)
        ax.legend(loc="upper right", fontsize=7.0)
    axes[0].set_ylabel("Amortized cost per query (USD, log scale)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"breakeven_flat_treatment.{ext}")
    print(f"wrote {FIG / 'breakeven_flat_treatment.png'} (+pdf)")


if __name__ == "__main__":
    main()
