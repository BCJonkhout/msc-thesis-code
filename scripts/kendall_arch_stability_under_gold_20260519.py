"""Kendall tau-b architecture rank stability under Codabench GOLD labels.

Same shape as ``kendall_arch_stability_rerun_20260519.py`` but the
per-(candidate, architecture) accuracy is taken from the Codabench
leaderboard log rather than the local consensus oracle. This is the
gold-validated counterpart to the 2026-05-19 rerun analysis.

Inputs (read-only):
  - ``outputs/sanity/codabench_vs_consensus_2026-05-16.json``
      Source for Flat + Naive RAG gold T/F per novel. The original
      submission used the 20-question calibration pool spanning B01,
      B08, B41, B50; this script slices to the 15-question pool
      (B01 + B08 + B50, B41 dropped) to match the rerun's denominator.
  - ``outputs/sanity/codabench_rerun_gold_20260519.json``
      Source for RAPTOR + GraphRAG gold T/F per novel (3 novels, 5
      questions each = 15 per arch). Produced by the parallel-batched
      submitter ``submit_rerun_to_codabench.py``.
  - ``outputs/sanity/novelqa_kendall_rerun_20260519.json``
      Consensus-oracle Kendall results for side-by-side comparison.

Decision rule (pre-registered Phase F.1):
  tau >= 0.67 AND >= 75% pairs STABLE   => context-divergence artefact
  tau in [0.33, 0.67)                    => mixed signal
  tau < 0.33                              => interaction is real

Outputs ``outputs/sanity/novelqa_kendall_under_gold_20260519.json``.

Provenance: PILOT-ERA. The held-out-gold (Codabench) stage of the
architecture-rank-stability tau-b series (consensus-oracle -> rescored ->
held-out-gold lineage); replaces the local consensus oracle with
Codabench gold T/F per question. Kept as a reproducibility record; not
the source of a current paper claim.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from pilot.cli.phase_f_kendall import _kendall_tau, _rank_by_f1

THRESHOLD = 0.67
LOW_BAND = 0.33

ARCHS_FULL = ("flat", "naive_rag", "raptor", "graphrag")
INCLUDED_NOVELS = ("B01", "B08", "B50")

CODE_ROOT = Path(__file__).resolve().parents[1]
MAY16_PATH = CODE_ROOT / "outputs" / "sanity" / "codabench_vs_consensus_2026-05-16.json"
RERUN_GOLD_PATH = CODE_ROOT / "outputs" / "sanity" / "codabench_rerun_gold_20260519.json"
CONSENSUS_KENDALL_PATH = (
    CODE_ROOT / "outputs" / "sanity" / "novelqa_kendall_rerun_20260519.json"
)
OUT_PATH = CODE_ROOT / "outputs" / "sanity" / "novelqa_kendall_under_gold_20260519.json"


# ──────────────────────────────────────────────────────────────────────
# Per-arch gold-accuracy extraction
# ──────────────────────────────────────────────────────────────────────

def _accuracy_from_slice_n15(per_novel: dict[str, Any]) -> dict[str, Any]:
    """Compute n_correct / n_questions on the B01+B08+B50 slice only.

    `per_novel` shape (Flat/Naive from May-16 file; RAPTOR/GraphRAG
    from rerun-gold file) is:
        { "B01": {"correctness_slice": "TTTFT", "correct": 4, "total": 5}, ... }
    Returns None when a target novel has no slice (Codabench missed it).
    """
    correct = 0
    total = 0
    for nid in INCLUDED_NOVELS:
        block = per_novel.get(nid)
        if not block or "correctness_slice" not in block:
            return {"n_correct": None, "n_questions": None, "accuracy_vs_gold": None,
                    "error": f"missing slice for {nid}"}
        slc = block["correctness_slice"]
        correct += sum(1 for c in slc if c == "T")
        total += sum(1 for c in slc if c in ("T", "F"))
    return {
        "n_correct": correct,
        "n_questions": total,
        "accuracy_vs_gold": round(correct / total, 6) if total else None,
    }


def _per_question_correct_vector(per_novel: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return [(novel_id, position_letter('T'/'F'/'?'), origin_key)] for jackknife."""
    out: list[tuple[str, str, str]] = []
    for nid in INCLUDED_NOVELS:
        block = per_novel.get(nid)
        if not block or "correctness_slice" not in block:
            continue
        slc = block["correctness_slice"]
        for pos_idx, c in enumerate(slc):
            out.append((nid, c, f"{nid}/{pos_idx}"))
    return out


def load_flat_naive_gold() -> dict[tuple[str, str], dict[str, Any]]:
    """Pull Flat + Naive RAG per-novel slices from the May-16 file.

    Returns {(candidate_short, arch): {"per_novel": {...}}} for the 9
    candidates present in the rerun grid. Sliced to B01+B08+B50.
    """
    data = json.loads(MAY16_PATH.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in data["submissions"]:
        cand = rec.get("candidate_short")
        arch = rec.get("architecture")
        if arch not in ("flat", "naive_rag"):
            continue
        if cand == "gemini-3.1-pro-preview":
            continue  # not in rerun grid
        scoring = rec.get("scoring") or {}
        per_novel_in = scoring.get("calibration_per_novel") or {}
        per_novel: dict[str, Any] = {}
        for nid in INCLUDED_NOVELS:
            blk = per_novel_in.get(nid)
            if blk and "correctness_slice" in blk:
                per_novel[nid] = {
                    "correctness_slice": blk["correctness_slice"],
                    "correct": blk.get("correct"),
                    "total": blk.get("total"),
                }
        out[(cand, arch)] = {
            "per_novel": per_novel,
            "submission_id": rec.get("submission_id"),
            "source": "codabench_vs_consensus_2026-05-16.json",
        }
    return out


def load_raptor_graphrag_gold() -> dict[tuple[str, str], dict[str, Any]]:
    """Pull RAPTOR + GraphRAG per-novel slices from the rerun-gold file."""
    data = json.loads(RERUN_GOLD_PATH.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in data["submissions"]:
        cand = rec.get("candidate")
        arch = rec.get("architecture")
        if arch not in ("raptor", "graphrag"):
            continue
        scoring = rec.get("scoring") or {}
        per_novel_in = scoring.get("per_novel") or {}
        per_novel: dict[str, Any] = {}
        for nid in INCLUDED_NOVELS:
            blk = per_novel_in.get(nid)
            if blk and "correctness_slice" in blk:
                per_novel[nid] = {
                    "correctness_slice": blk["correctness_slice"],
                    "correct": blk.get("correct"),
                    "total": blk.get("total"),
                }
        out[(cand, arch)] = {
            "per_novel": per_novel,
            "submission_id": rec.get("submission_id"),
            "source": "codabench_rerun_gold_20260519.json",
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# Kendall computation
# ──────────────────────────────────────────────────────────────────────

def _rankings_from_scores(score_table: dict[str, dict[str, float]],
                          archs: tuple[str, ...]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cand, scores in score_table.items():
        if any(scores.get(a) is None for a in archs):
            continue
        out[cand] = _rank_by_f1({a: scores[a] for a in archs})
    return out


def _kendall_block(score_table: dict[str, dict[str, float]],
                   archs: tuple[str, ...], label: str) -> dict[str, Any]:
    rankings = _rankings_from_scores(score_table, archs)
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
        "score_table": {c: {a: score_table[c][a] for a in archs} for c in rankings},
        "pairs": pair_results,
    }


def _band(tau: float | None) -> str:
    if tau is None:
        return "unknown"
    if tau >= THRESHOLD:
        return "context-divergence-artefact (>=0.67)"
    if tau >= LOW_BAND:
        return "mixed-signal (>=0.33, <0.67)"
    return "interaction-real (<0.33)"


# ──────────────────────────────────────────────────────────────────────
# Jackknife on the 15 questions
# ──────────────────────────────────────────────────────────────────────

def _per_question_records(
    per_arch_per_cand: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, str | None]]:
    """Index: (cand, arch, qkey) -> 'T'/'F'/None.

    qkey is 'B01/0' .. 'B50/4' — fixed positions within the 5-question
    per-novel calibration slice (in calibration_pool.jsonl order).
    """
    out: dict[tuple[str, str, str], dict[str, str | None]] = {}
    for (cand, arch), payload in per_arch_per_cand.items():
        for nid in INCLUDED_NOVELS:
            blk = payload["per_novel"].get(nid)
            if not blk or "correctness_slice" not in blk:
                continue
            for pos_idx, c in enumerate(blk["correctness_slice"]):
                qkey = f"{nid}/{pos_idx}"
                if c not in ("T", "F"):
                    continue
                out[(cand, arch, qkey)] = c  # type: ignore[assignment]
    return out


def _question_keys() -> list[str]:
    return [f"{n}/{i}" for n in INCLUDED_NOVELS for i in range(5)]


def _scores_excluding(per_q: dict, cands: list[str], archs: tuple[str, ...],
                      excluded: str | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    qkeys = [q for q in _question_keys() if q != excluded]
    for cand in cands:
        d: dict[str, float] = {}
        for arch in archs:
            t = n = 0
            for qk in qkeys:
                v = per_q.get((cand, arch, qk))
                if v == "T":
                    t += 1
                    n += 1
                elif v == "F":
                    n += 1
            d[arch] = round(t / n, 6) if n else None  # type: ignore[assignment]
        out[cand] = d
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    flat_naive = load_flat_naive_gold()
    raptor_graphrag = load_raptor_graphrag_gold()
    all_records: dict[tuple[str, str], dict[str, Any]] = {**flat_naive, **raptor_graphrag}

    # 1) Per-(cand, arch) accuracy table under gold (n=15)
    per_cand_per_arch: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (cand, arch), payload in all_records.items():
        acc_block = _accuracy_from_slice_n15(payload["per_novel"])
        per_cand_per_arch[cand][arch] = {
            **acc_block,
            "submission_id": payload.get("submission_id"),
            "source": payload.get("source"),
            "per_novel_slice": payload["per_novel"],
        }

    # 2) Per-architecture mean accuracy under gold
    per_arch_accuracies: dict[str, list[float]] = defaultdict(list)
    for cand, arch_blk in per_cand_per_arch.items():
        for arch, blk in arch_blk.items():
            v = blk.get("accuracy_vs_gold")
            if isinstance(v, (int, float)):
                per_arch_accuracies[arch].append(v)
    per_arch_mean = {
        arch: round(statistics.fmean(vs), 4) if vs else None
        for arch, vs in per_arch_accuracies.items()
    }
    per_arch_median = {
        arch: round(statistics.median(vs), 4) if vs else None
        for arch, vs in per_arch_accuracies.items()
    }
    arch_ranking_by_mean = sorted(
        ARCHS_FULL, key=lambda a: per_arch_mean.get(a) or -1, reverse=True
    )

    # 3) Per-candidate architecture ranking under gold
    score_table: dict[str, dict[str, float]] = {}
    per_candidate_ranking: dict[str, list[str]] = {}
    for cand, arch_blk in per_cand_per_arch.items():
        scores: dict[str, float] = {}
        for arch in ARCHS_FULL:
            v = arch_blk.get(arch, {}).get("accuracy_vs_gold")
            if isinstance(v, (int, float)):
                scores[arch] = v
        score_table[cand] = scores
        if len(scores) == len(ARCHS_FULL):
            per_candidate_ranking[cand] = _rank_by_f1(scores)

    # 4) Pairwise Kendall tau-b on 4-arch accuracy vectors
    block_4arch = _kendall_block(score_table, ARCHS_FULL, "gold, 4-arch")
    band_4arch = _band(block_4arch["median_tau"])

    # 5) Per-question jackknife on 4-arch median tau-b under gold
    per_q = _per_question_records(all_records)
    cands_with_all_archs = list(per_candidate_ranking.keys())
    qkeys = _question_keys()
    base_scores = _scores_excluding(per_q, cands_with_all_archs, ARCHS_FULL, None)
    base_block = _kendall_block(base_scores, ARCHS_FULL, "jackknife baseline (no drop)")
    jk_records: list[dict[str, Any]] = []
    for qk in qkeys:
        scores = _scores_excluding(per_q, cands_with_all_archs, ARCHS_FULL, qk)
        blk = _kendall_block(scores, ARCHS_FULL, f"jk drop {qk}")
        jk_records.append({
            "dropped_question_position": qk,
            "n_candidates_ranked": blk["n_candidates"],
            "median_tau": blk["median_tau"],
            "mean_tau": blk["mean_tau"],
            "stable_fraction": blk["stable_fraction"],
            "n_stable": blk["n_stable"],
            "n_pairs": blk["n_pairs"],
            "band": _band(blk["median_tau"]),
        })
    medians = [r["median_tau"] for r in jk_records if r["median_tau"] is not None]
    sfs = [r["stable_fraction"] for r in jk_records if r["stable_fraction"] is not None]
    base_band = _band(base_block["median_tau"])
    n_flip = sum(1 for r in jk_records if r["band"] != base_band)
    jackknife = {
        "baseline": {
            "median_tau": base_block["median_tau"],
            "mean_tau": base_block["mean_tau"],
            "stable_fraction": base_block["stable_fraction"],
            "n_stable": base_block["n_stable"],
            "n_pairs": base_block["n_pairs"],
            "band": base_band,
        },
        "summary": {
            "n_questions_dropped_one_at_a_time": len(jk_records),
            "median_tau_min": round(min(medians), 4) if medians else None,
            "median_tau_median": round(statistics.median(medians), 4) if medians else None,
            "median_tau_max": round(max(medians), 4) if medians else None,
            "median_tau_mean": round(statistics.fmean(medians), 4) if medians else None,
            "stable_fraction_min": round(min(sfs), 4) if sfs else None,
            "stable_fraction_max": round(max(sfs), 4) if sfs else None,
            "n_drops_that_flip_baseline_band": n_flip,
            "baseline_band": base_band,
        },
        "per_dropped_question_position": jk_records,
    }

    # 6) Side-by-side with consensus-based numbers
    cons = json.loads(CONSENSUS_KENDALL_PATH.read_text(encoding="utf-8"))
    cons_primary = cons.get("blocks", {}).get("full__4arch", {})
    side_by_side = {
        "consensus_oracle_2026-05-19": {
            "median_tau": cons.get("primary_median_tau"),
            "stable_fraction": cons.get("primary_stable_fraction"),
            "band": cons.get("primary_decision_band"),
            "n_candidates": cons.get("n_candidates"),
        },
        "gold_2026-05-19": {
            "median_tau": block_4arch["median_tau"],
            "stable_fraction": block_4arch["stable_fraction"],
            "band": band_4arch,
            "n_candidates": block_4arch["n_candidates"],
        },
        "band_change": (
            cons.get("primary_decision_band") != band_4arch
        ),
        "delta_median_tau": (
            round(block_4arch["median_tau"] - cons.get("primary_median_tau"), 4)
            if block_4arch["median_tau"] is not None
            and cons.get("primary_median_tau") is not None
            else None
        ),
    }

    # Pretty per-cand-per-arch accuracy table (rounded)
    pretty_table: dict[str, dict[str, float | None]] = {}
    for cand in sorted(per_cand_per_arch.keys()):
        pretty_table[cand] = {
            arch: per_cand_per_arch[cand].get(arch, {}).get("accuracy_vs_gold")
            for arch in ARCHS_FULL
        }

    out: dict[str, Any] = {
        "schema_version": 1,
        "threshold": THRESHOLD,
        "low_band": LOW_BAND,
        "rerun_date": "2026-05-19",
        "included_novels": list(INCLUDED_NOVELS),
        "n_questions_in_slice": 15,
        "primary_decision_band_under_gold": band_4arch,
        "primary_median_tau_under_gold": block_4arch["median_tau"],
        "primary_stable_fraction_under_gold": block_4arch["stable_fraction"],
        "n_candidates": block_4arch["n_candidates"],
        "per_candidate_per_arch_accuracy": pretty_table,
        "per_candidate_per_arch_full": per_cand_per_arch,
        "per_candidate_arch_ranking": per_candidate_ranking,
        "per_architecture_mean_accuracy": per_arch_mean,
        "per_architecture_median_accuracy": per_arch_median,
        "architecture_ranking_by_mean": list(arch_ranking_by_mean),
        "kendall_full_4arch": block_4arch,
        "jackknife_full_4arch": jackknife,
        "side_by_side_vs_consensus": side_by_side,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Kendall tau-b UNDER GOLD — RERUN 2026-05-19 (4-arch, n=15)")
    print(f"  threshold for STABLE: tau >= {THRESHOLD}\n")
    print(f"primary median tau (gold): {block_4arch['median_tau']}")
    print(f"primary stable fraction:   {block_4arch['stable_fraction']}")
    print(f"primary band:              {band_4arch}\n")
    print("per-arch mean accuracy under gold:")
    for arch in ARCHS_FULL:
        print(f"  {arch:>10s}: {per_arch_mean[arch]} (median {per_arch_median[arch]})")
    print(f"\narch ranking by mean: {arch_ranking_by_mean}")
    print("\nside-by-side:")
    print(f"  consensus  median tau = {side_by_side['consensus_oracle_2026-05-19']['median_tau']}, "
          f"band = {side_by_side['consensus_oracle_2026-05-19']['band']}")
    print(f"  gold       median tau = {side_by_side['gold_2026-05-19']['median_tau']}, "
          f"band = {side_by_side['gold_2026-05-19']['band']}")
    print(f"  band change: {side_by_side['band_change']}")
    print(f"\nper-(cand, arch) accuracy table under gold (n=15):")
    print(f"  {'candidate':<32s} {'flat':>6s} {'naive':>6s} {'raptor':>7s} {'graphrag':>9s}")
    for cand in sorted(pretty_table.keys()):
        row = pretty_table[cand]
        cells = [f"{row[a]:.4f}" if isinstance(row[a], (int, float)) else "  -   "
                 for a in ARCHS_FULL]
        print(f"  {cand:<32s} {cells[0]:>6s} {cells[1]:>6s} {cells[2]:>7s} {cells[3]:>9s}")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
