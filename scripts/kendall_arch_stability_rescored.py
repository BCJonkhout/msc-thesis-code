"""Kendall tau-b architecture rank stability — rescored Phase G NovelQA.

Reads ``outputs/sanity/novelqa_local_scores_rescored.json`` and emits
``outputs/sanity/novelqa_kendall_rescored.json``.

Computes TWO Kendall analyses for each consensus variant (full and LOO):

  1. Full 4-arch tau-b — matches the original headline finding
     methodology (flat, naive_rag, raptor, graphrag). Only candidates
     with all 4 architectures scored contribute.

  2. Flat + Naive RAG only tau-b — the "clean" comparison where the
     context fed to the answerer is verified identical across
     candidates (20 distinct prompt_hashes shared across all
     candidates). RAPTOR and GraphRAG diverge per candidate (their
     retrieved/clustered context is itself produced by the candidate
     under test), so any rank instability that involves them is
     confounded by context divergence rather than answerer-style.

     For a 2-architecture ranking tau-b takes only 3 values:
        +1  identical order, -1  reversed order, 0  tie
     (a tie occurs when a candidate has the same accuracy on flat and
     naive_rag — in tau-b ties count as neither concordant nor
     discordant). The decision rule stays at tau >= 0.67, so on this
     2-arch axis a candidate-pair is STABLE iff both candidates pick
     the same architecture as the better one (tau == +1).

Decision threshold: tau >= 0.67 (pre-registered in Phase F.1).
"""
from __future__ import annotations

import json
import statistics
from itertools import combinations
from pathlib import Path

from pilot.cli.phase_f_kendall import _kendall_tau, _rank_by_f1


THRESHOLD = 0.67


def _rankings(per_cand: dict, archs: tuple[str, ...]) -> tuple[
    dict[str, list[str]], dict[str, dict[str, float]]
]:
    rankings: dict[str, list[str]] = {}
    score_table: dict[str, dict[str, float]] = {}
    for cand, by_arch in per_cand.items():
        scores: dict[str, float] = {}
        for a in archs:
            v = by_arch.get(a, {}).get("accuracy_vs_consensus")
            if isinstance(v, (int, float)):
                scores[a] = v
        if len(scores) < len(archs):
            continue
        rankings[cand] = _rank_by_f1(scores)
        score_table[cand] = scores
    return rankings, score_table


def _kendall_block(per_cand: dict, archs: tuple[str, ...], label: str) -> dict:
    rankings, score_table = _rankings(per_cand, archs)
    pairs = list(combinations(sorted(rankings.keys()), 2))
    taus = []
    pair_results = []
    n_stable = 0
    for a, b in pairs:
        tau, conc, disc = _kendall_tau(rankings[a], rankings[b])
        stable = tau >= THRESHOLD
        if stable:
            n_stable += 1
        taus.append(tau)
        pair_results.append({
            "a": a, "b": b,
            "tau": round(tau, 4),
            "concordant": conc, "discordant": disc,
            "stable": stable,
        })
    median = round(statistics.median(taus), 4) if taus else None
    mean = round(statistics.fmean(taus), 4) if taus else None
    return {
        "label": label,
        "archs": list(archs),
        "n_candidates": len(rankings),
        "n_pairs": len(pairs),
        "n_stable": n_stable,
        "stable_fraction": round(n_stable / len(pairs), 4) if pairs else None,
        "median_tau": median,
        "mean_tau": mean,
        "rankings": rankings,
        "score_table": score_table,
        "pairs": pair_results,
    }


def main() -> int:
    code_root = Path(__file__).resolve().parents[1]
    in_path = code_root / "outputs" / "sanity" / "novelqa_local_scores_rescored.json"
    out_path = code_root / "outputs" / "sanity" / "novelqa_kendall_rescored.json"
    data = json.loads(in_path.read_text(encoding="utf-8"))

    archs_full = ("flat", "naive_rag", "raptor", "graphrag")
    archs_clean = ("flat", "naive_rag")

    blocks: dict[str, dict] = {}
    for variant in ("full", "loo"):
        per_cand = data[f"per_candidate_{variant}"]
        blocks[f"{variant}__4arch"] = _kendall_block(
            per_cand, archs_full, f"{variant} consensus, 4-arch"
        )
        blocks[f"{variant}__flat_naive"] = _kendall_block(
            per_cand, archs_clean, f"{variant} consensus, flat+naive_rag only"
        )

    out = {
        "schema_version": 1,
        "threshold": THRESHOLD,
        "blocks": blocks,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Stdout summary
    print(f"Kendall tau-b architecture stability (RESCORED Phase G NovelQA)")
    print(f"  threshold for STABLE: tau >= {THRESHOLD}\n")
    print(f"{'block':<32} {'n_cand':>6} {'pairs':>6} "
          f"{'stable':>7} {'stable%':>8} {'med_tau':>8} {'mean_tau':>9}")
    print("-" * 90)
    for key, b in blocks.items():
        sf = b["stable_fraction"]
        sf_str = f"{100 * sf:.1f}%" if sf is not None else "  -  "
        print(f"{key:<32} {b['n_candidates']:>6} {b['n_pairs']:>6} "
              f"{b['n_stable']:>7} {sf_str:>8} "
              f"{(b['median_tau'] if b['median_tau'] is not None else 0):>8.3f} "
              f"{(b['mean_tau'] if b['mean_tau'] is not None else 0):>9.3f}")
    print(f"\n[kendall_arch_stability_rescored] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
