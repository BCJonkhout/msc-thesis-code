"""Cross-dataset Kendall tau-b on the 9-candidate x 4-arch grid.

Combines:
  - NovelQA gold accuracy (15-question slice; from
    ``outputs/sanity/novelqa_kendall_under_gold_20260519.json``).
  - QASPER gold Answer-F1 (20-question pool; from
    ``outputs/sanity/qasper_rerun_gold_20260520.json``).

Three rank-stability views:
  1. NovelQA-only tau-b : median Kendall tau-b across the 36 candidate
                          pairs on the 4-arch NovelQA accuracy vector.
                          (Should match the existing 0.6667 value;
                          recomputed here to keep this file self-
                          contained.)
  2. QASPER-only tau-b  : same shape on the QASPER Answer-F1 vector.
  3. Cross-dataset tau-b: concatenate per-candidate (QASPER_4 ||
                          NovelQA_4) to an 8-dim accuracy vector, then
                          rank each candidate's 8 cells and take
                          pairwise Kendall tau-b. This tests whether
                          two candidates that produce similar full-grid
                          orderings of arches & datasets agree.

The honest-framing alternative is also reported: side-by-side
NovelQA-only vs QASPER-only median tau-b. If both land in the same
band (mixed-signal [0.33, 0.67) or context-divergence (>= 0.67)) the
cross-dataset claim is supported. If they diverge sharply the
divergence is reported as the finding.

Output: ``outputs/sanity/kendall_cross_dataset_under_gold_20260520.json``

Provenance: PILOT-ERA. Computes the cross-dataset (QASPER || NovelQA)
tau-b under held-out gold. The tau-b material this produces backs the
paper's ``tab:results-tau`` in the pilot-licensing appendix, so it is the
documented source for that single appendix table; it is otherwise off the
canonical main-study path. Kept as a reproducibility record.
"""
from __future__ import annotations

import json
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from pilot.cli.phase_f_kendall import _kendall_tau, _rank_by_f1  # noqa: E402

THRESHOLD = 0.67
LOW_BAND = 0.33
ARCHS = ("flat", "naive_rag", "raptor", "graphrag")

SANITY = CODE_ROOT / "outputs" / "sanity"
NOVELQA_GOLD = SANITY / "novelqa_kendall_under_gold_20260519.json"
QASPER_GOLD = SANITY / "qasper_rerun_gold_20260520.json"
OUT_PATH = SANITY / "kendall_cross_dataset_under_gold_20260520.json"


def _band(tau: float | None) -> str:
    if tau is None:
        return "unknown"
    if tau >= THRESHOLD:
        return "context-divergence-artefact (>=0.67)"
    if tau >= LOW_BAND:
        return "mixed-signal (>=0.33, <0.67)"
    return "interaction-real (<0.33)"


def _kendall_block_on_scores(
    score_table: dict[str, dict[str, float]],
    archs: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    """Pairwise Kendall tau-b across candidates given a (cand -> arch ->
    score) table.

    Skips candidates that are missing any of ``archs``."""
    rankings: dict[str, list[str]] = {}
    cleaned_scores: dict[str, dict[str, float]] = {}
    for cand, scores in score_table.items():
        if any(scores.get(a) is None for a in archs):
            continue
        clean = {a: float(scores[a]) for a in archs}
        rankings[cand] = _rank_by_f1(clean)
        cleaned_scores[cand] = clean

    pairs = list(combinations(sorted(rankings.keys()), 2))
    taus: list[float] = []
    pair_results: list[dict[str, Any]] = []
    n_stable = 0
    for a, b in pairs:
        tau, conc, disc = _kendall_tau(rankings[a], rankings[b])
        stable = tau >= THRESHOLD
        if stable:
            n_stable += 1
        taus.append(tau)
        pair_results.append({
            "a": a, "b": b, "tau": round(tau, 4),
            "concordant": conc, "discordant": disc, "stable": stable,
        })
    return {
        "label": label,
        "archs": list(archs),
        "n_candidates": len(rankings),
        "n_pairs": len(pairs),
        "n_stable": n_stable,
        "stable_fraction": round(n_stable / len(pairs), 4) if pairs else None,
        "median_tau": round(statistics.median(taus), 4) if taus else None,
        "mean_tau": round(statistics.fmean(taus), 4) if taus else None,
        "rankings": rankings,
        "score_table": cleaned_scores,
        "pairs": pair_results,
    }


def _cross_dataset_block(
    novelqa: dict[str, dict[str, float]],
    qasper: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Concatenate the (QASPER_4 || NovelQA_4) 8-dim vectors per
    candidate, then compute pairwise Kendall tau-b on the resulting
    ranking of 8 (arch, dataset) cells.

    Each cell label is ``"<arch>__<dataset>"``. Candidates missing any
    of the 8 cells are dropped from the comparison.
    """
    archs_full: tuple[str, ...] = tuple(
        [f"{a}__qasper" for a in ARCHS] + [f"{a}__novelqa" for a in ARCHS]
    )
    score_table: dict[str, dict[str, float]] = {}
    candidates = sorted(set(novelqa.keys()) & set(qasper.keys()))
    for cand in candidates:
        n = novelqa.get(cand, {})
        q = qasper.get(cand, {})
        row: dict[str, float] = {}
        ok = True
        for arch in ARCHS:
            v_q = q.get(arch)
            v_n = n.get(arch)
            if not isinstance(v_q, (int, float)) or not isinstance(v_n, (int, float)):
                ok = False
                break
            row[f"{arch}__qasper"] = float(v_q)
            row[f"{arch}__novelqa"] = float(v_n)
        if ok:
            score_table[cand] = row
    return _kendall_block_on_scores(
        score_table, archs_full, "cross-dataset, 8-cell (QASPER_4 || NovelQA_4)"
    )


def main() -> int:
    if not NOVELQA_GOLD.exists():
        raise SystemExit(f"missing NovelQA gold file: {NOVELQA_GOLD}")
    if not QASPER_GOLD.exists():
        raise SystemExit(f"missing QASPER gold file: {QASPER_GOLD}")

    novelqa = json.loads(NOVELQA_GOLD.read_text(encoding="utf-8"))
    qasper = json.loads(QASPER_GOLD.read_text(encoding="utf-8"))

    novelqa_table: dict[str, dict[str, float]] = novelqa[
        "per_candidate_per_arch_accuracy"
    ]
    qasper_table: dict[str, dict[str, float]] = qasper[
        "per_candidate_per_arch_answer_f1"
    ]

    common = sorted(set(novelqa_table) & set(qasper_table))

    # Per-dataset tau-b on the same candidate intersection so the
    # side-by-side comparison is apples-to-apples.
    nq_scores = {c: novelqa_table[c] for c in common}
    qa_scores = {c: qasper_table[c] for c in common}

    novelqa_block = _kendall_block_on_scores(
        nq_scores, ARCHS, "NovelQA-only (gold, 4-arch, n=15)"
    )
    qasper_block = _kendall_block_on_scores(
        qa_scores, ARCHS, "QASPER-only (gold, 4-arch, n=20)"
    )
    cross_block = _cross_dataset_block(nq_scores, qa_scores)

    nq_med = novelqa_block["median_tau"]
    qa_med = qasper_block["median_tau"]
    cx_med = cross_block["median_tau"]

    nq_band = _band(nq_med)
    qa_band = _band(qa_med)
    cx_band = _band(cx_med)

    # Cross-dataset finding logic. Strengthen / weaken / refine vs the
    # NovelQA-only headline are pre-registered here for clarity.
    headline = "indeterminate"
    if nq_med is not None and qa_med is not None:
        if nq_band == qa_band:
            headline = (
                f"AGREEMENT: NovelQA tau-b={nq_med} (band={nq_band}) ~= "
                f"QASPER tau-b={qa_med} (band={qa_band}); cross-dataset "
                f"claim STRENGTHENED."
            )
        elif (qa_med is not None and nq_med is not None
              and abs(qa_med - nq_med) < 0.15):
            headline = (
                f"NEAR-AGREEMENT: NovelQA tau-b={nq_med} (band={nq_band}) "
                f"vs QASPER tau-b={qa_med} (band={qa_band}); same direction, "
                f"different band; REFINE the headline."
            )
        else:
            headline = (
                f"DIVERGENCE: NovelQA tau-b={nq_med} (band={nq_band}) "
                f"vs QASPER tau-b={qa_med} (band={qa_band}); "
                f"cross-dataset claim WEAKENED."
            )

    out: dict[str, Any] = {
        "schema_version": 1,
        "analysis_date": "2026-05-20",
        "threshold": THRESHOLD,
        "low_band": LOW_BAND,
        "n_candidates_common": len(common),
        "candidates_common": common,
        "inputs": {
            "novelqa_gold": str(NOVELQA_GOLD.relative_to(CODE_ROOT)),
            "qasper_gold": str(QASPER_GOLD.relative_to(CODE_ROOT)),
            "novelqa_n_questions": novelqa.get("n_questions_in_slice"),
            "qasper_n_questions": qasper.get("calibration_pool_size"),
        },
        "per_dataset": {
            "novelqa": {
                "median_tau": nq_med,
                "mean_tau": novelqa_block["mean_tau"],
                "stable_fraction": novelqa_block["stable_fraction"],
                "band": nq_band,
                "n_pairs": novelqa_block["n_pairs"],
            },
            "qasper": {
                "median_tau": qa_med,
                "mean_tau": qasper_block["mean_tau"],
                "stable_fraction": qasper_block["stable_fraction"],
                "band": qa_band,
                "n_pairs": qasper_block["n_pairs"],
            },
        },
        "cross_dataset": {
            "median_tau": cx_med,
            "mean_tau": cross_block["mean_tau"],
            "stable_fraction": cross_block["stable_fraction"],
            "band": cx_band,
            "n_pairs": cross_block["n_pairs"],
            "n_cells": len(cross_block["archs"]),
        },
        "headline": headline,
        "blocks": {
            "novelqa_only": novelqa_block,
            "qasper_only": qasper_block,
            "cross_dataset_8cell": cross_block,
        },
        "per_candidate_rankings": {
            "novelqa": novelqa_block["rankings"],
            "qasper": qasper_block["rankings"],
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nCross-dataset Kendall tau-b (2026-05-20):")
    print(f"  n_candidates (intersection): {len(common)}")
    print(f"  NovelQA-only      median tau-b = {nq_med}  ({nq_band})")
    print(f"  QASPER-only       median tau-b = {qa_med}  ({qa_band})")
    print(f"  cross-dataset 8c  median tau-b = {cx_med}  ({cx_band})")
    print(f"\n  headline: {headline}")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
