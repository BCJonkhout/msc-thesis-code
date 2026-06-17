r"""Bootstrap 95% CI, permutation p-value, and Monte-Carlo power
estimate for the per-dataset and cross-dataset median Kendall tau-b.

For each of the three primary tau-b views (NovelQA-only, QASPER-only,
cross-dataset 8-cell), we report:

  - Bootstrap 95% CI on the median tau-b across the 36 candidate
    pairs. We resample the 36 pairs with replacement N_BOOT=10,000
    times; the 2.5th and 97.5th percentiles of the bootstrap
    distribution form the CI. This is the standard pair-level
    bootstrap; it captures sampling variability of the candidate-pair
    set, holding the underlying rankings fixed.

  - One-sided permutation p-value for H0: "candidate-architecture
    interaction labels are exchangeable" (no real rank structure).
    For each of N_PERM=10,000 permutations we:
        for each candidate independently, shuffle its 4-cell
        (or 8-cell, cross-dataset) score vector across architectures;
        re-derive the per-candidate ranking; re-compute median tau-b.
    Under H0 the candidate's ranking is a random permutation, so
    pairwise tau-b should fluctuate around 0. p-value = fraction of
    permuted median tau-b values >= observed (one-sided test for
    higher-than-random rank agreement).

  - Monte-Carlo power estimate: under the alternative "true population
    tau = 0.5" with N_CAND=9 candidates and N_Q questions per cell, what
    is the probability the observed permutation p-value lands below
    alpha=0.05? We simulate by perturbing a fixed canonical ranking
    with noise calibrated so the expected pairwise tau-b equals 0.5,
    then run the permutation test on each simulated dataset.
    Approximate but informative for thesis writeup.

Citation: \cite{dror2018hitchhiker} for the methodology of significance
testing in NLP comparisons.

Output: ``outputs/sanity/kendall_stats_20260520.json``

Provenance: PILOT-ERA. Bootstrap CI / permutation test / Monte-Carlo
power for the cross-dataset tau-b above; together they back the paper's
``tab:results-tau`` in the pilot-licensing appendix (the documented source
for that single appendix table) and are otherwise off the canonical
main-study path. Kept as a reproducibility record.
"""
from __future__ import annotations

import json
import math
import random
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
CROSS_PATH = SANITY / "kendall_cross_dataset_under_gold_20260520.json"
OUT_PATH = SANITY / "kendall_stats_20260520.json"

N_BOOT = 10_000
N_PERM = 10_000
# Power analysis budget is deliberately smaller than the headline-
# stat budget because each simulation itself runs N_POWER_PERMS
# permutations, so the total cost is N_POWER_SIMS x N_POWER_PERMS
# permutation evaluations. 500 x 500 = 2.5e5 evaluations is enough
# resolution for a power point-estimate at the 0.02 absolute level
# and keeps the script under ~3 min on a single core.
N_POWER_SIMS = 500
N_POWER_PERMS = 500
ALPHA = 0.05
TRUE_TAU_FOR_POWER = 0.5
SEED = 20260520


# ──────────────────────────────────────────────────────────────────────
# Pair-level metrics from score table
# ──────────────────────────────────────────────────────────────────────

def _rankings_from(score_table: dict[str, dict[str, float]],
                   archs: tuple[str, ...]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cand, scores in score_table.items():
        if any(scores.get(a) is None for a in archs):
            continue
        out[cand] = _rank_by_f1({a: float(scores[a]) for a in archs})
    return out


def _pairwise_taus(rankings: dict[str, list[str]]) -> list[float]:
    out: list[float] = []
    for a, b in combinations(sorted(rankings.keys()), 2):
        tau, _, _ = _kendall_tau(rankings[a], rankings[b])
        out.append(tau)
    return out


def _median_tau_from_scores(score_table: dict[str, dict[str, float]],
                            archs: tuple[str, ...]) -> float | None:
    rk = _rankings_from(score_table, archs)
    taus = _pairwise_taus(rk)
    if not taus:
        return None
    return statistics.median(taus)


# ──────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────────────────────────

def bootstrap_median_ci(taus: list[float],
                        n_boot: int = N_BOOT,
                        seed: int = SEED,
                        alpha: float = ALPHA) -> dict[str, Any]:
    if not taus:
        return {"median_tau": None, "ci_low": None, "ci_high": None,
                "n_pairs": 0, "n_boot": n_boot}
    rng = random.Random(seed)
    medians: list[float] = []
    n = len(taus)
    for _ in range(n_boot):
        sample = [taus[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo_idx = int(math.floor((alpha / 2.0) * n_boot))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1
    hi_idx = max(0, min(n_boot - 1, hi_idx))
    return {
        "median_tau": round(statistics.median(taus), 4),
        "ci_low": round(medians[lo_idx], 4),
        "ci_high": round(medians[hi_idx], 4),
        "ci_alpha": alpha,
        "n_pairs": n,
        "n_boot": n_boot,
    }


# ──────────────────────────────────────────────────────────────────────
# Permutation test
# ──────────────────────────────────────────────────────────────────────

def _permute_within_candidate(score_table: dict[str, dict[str, float]],
                              archs: tuple[str, ...],
                              rng: random.Random) -> dict[str, dict[str, float]]:
    """Shuffle each candidate's score vector across ``archs`` independently."""
    out: dict[str, dict[str, float]] = {}
    for cand, scores in score_table.items():
        if any(scores.get(a) is None for a in archs):
            out[cand] = scores
            continue
        vals = [float(scores[a]) for a in archs]
        rng.shuffle(vals)
        out[cand] = {a: vals[i] for i, a in enumerate(archs)}
    return out


def permutation_pvalue(score_table: dict[str, dict[str, float]],
                       archs: tuple[str, ...],
                       n_perm: int = N_PERM,
                       seed: int = SEED) -> dict[str, Any]:
    observed = _median_tau_from_scores(score_table, archs)
    if observed is None:
        return {"observed_median_tau": None, "p_value_one_sided": None,
                "n_perm": n_perm}
    rng = random.Random(seed + 1)
    n_ge = 0
    perm_medians: list[float] = []
    for _ in range(n_perm):
        permed = _permute_within_candidate(score_table, archs, rng)
        m = _median_tau_from_scores(permed, archs)
        if m is None:
            continue
        perm_medians.append(m)
        if m >= observed:
            n_ge += 1
    # Add-one smoothing to keep the p-value bounded away from 0 (a
    # standard recommendation in NLP significance testing - dror18).
    pval = (n_ge + 1) / (len(perm_medians) + 1)
    return {
        "observed_median_tau": round(observed, 4),
        "p_value_one_sided": round(pval, 6),
        "n_perm_effective": len(perm_medians),
        "n_perm": n_perm,
        "perm_median_mean": round(statistics.fmean(perm_medians), 4)
            if perm_medians else None,
        "perm_median_median": round(statistics.median(perm_medians), 4)
            if perm_medians else None,
        "perm_median_p95": round(_quantile(perm_medians, 0.95), 4)
            if perm_medians else None,
        "smoothing": "add-one (Dror et al. 2018)",
    }


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[idx]


# ──────────────────────────────────────────────────────────────────────
# Power analysis (Monte Carlo, simulated rankings under true_tau=0.5)
# ──────────────────────────────────────────────────────────────────────

def _ranking_with_target_tau(reference: list[str], target_tau: float,
                             rng: random.Random) -> list[str]:
    """Produce a ranking that has expected Kendall tau-b vs the
    ``reference`` ranking equal to ``target_tau``.

    Strategy: pairwise. For each (i, j) pair in the reference, with
    probability ``p = (1 + target_tau) / 2`` keep the reference order,
    else flip it. Then build a ranking consistent with as many of these
    pairwise constraints as possible via a greedy tournament sort.
    This is approximate (constraint set may be inconsistent) but is
    good enough for the Monte-Carlo power estimate at the granularity
    we need.
    """
    n = len(reference)
    if n <= 1:
        return list(reference)
    p_keep = (1.0 + target_tau) / 2.0
    # Build a score per item from pairwise preferences.
    wins: dict[str, int] = {x: 0 for x in reference}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = reference[i], reference[j]
            keep = rng.random() < p_keep
            if keep:
                wins[a] += 1
            else:
                wins[b] += 1
    # Tie-break by reference order so the ranking is deterministic
    # given the wins.
    return sorted(reference, key=lambda x: (-wins[x], reference.index(x)))


def _simulate_scoretable(canonical: list[str],
                         n_candidates: int,
                         target_tau: float,
                         rng: random.Random) -> dict[str, dict[str, float]]:
    """Build a synthetic score table whose candidate-pair rankings have
    expected pairwise Kendall tau-b ~= target_tau vs each other.

    Each candidate gets a noisy ranking around ``canonical``; scores
    are assigned by rank position so the downstream Kendall code can
    use the same helpers.
    """
    archs = canonical
    out: dict[str, dict[str, float]] = {}
    for k in range(n_candidates):
        ranking = _ranking_with_target_tau(archs, target_tau, rng)
        # Assign scores by rank position (1.0, 0.75, 0.5, 0.25, ...)
        out[f"cand_{k}"] = {
            a: 1.0 - (i / max(1, len(archs) - 1))
            for i, a in enumerate(ranking)
        }
    return out


def power_estimate(archs: tuple[str, ...],
                   n_candidates: int,
                   true_tau: float = TRUE_TAU_FOR_POWER,
                   n_sims: int = N_POWER_SIMS,
                   n_perm: int = N_POWER_PERMS,
                   alpha: float = ALPHA,
                   seed: int = SEED) -> dict[str, Any]:
    """Estimate Pr(p < alpha | true median tau-b ~= true_tau).

    Each simulation draws a fresh synthetic score table whose pairwise
    Kendall tau-b is centred at ``true_tau``, runs the permutation
    test at the cheaper ``n_perm`` budget, and counts the fraction of
    sims that reject H0.
    """
    rng = random.Random(seed + 2)
    canonical = list(archs)
    n_reject = 0
    realised_taus: list[float] = []
    realised_pvals: list[float] = []
    for _ in range(n_sims):
        table = _simulate_scoretable(canonical, n_candidates, true_tau, rng)
        observed = _median_tau_from_scores(table, archs)
        if observed is None:
            continue
        realised_taus.append(observed)
        rng_inner = random.Random(rng.random())
        n_ge = 0
        eff = 0
        for _p in range(n_perm):
            permed = _permute_within_candidate(table, archs, rng_inner)
            m = _median_tau_from_scores(permed, archs)
            if m is None:
                continue
            eff += 1
            if m >= observed:
                n_ge += 1
        pval = (n_ge + 1) / (eff + 1)
        realised_pvals.append(pval)
        if pval < alpha:
            n_reject += 1
    return {
        "true_tau_for_power": true_tau,
        "n_candidates": n_candidates,
        "n_archs": len(archs),
        "n_sims": n_sims,
        "n_perm_per_sim": n_perm,
        "alpha": alpha,
        "power": round(n_reject / max(1, len(realised_pvals)), 4)
            if realised_pvals else None,
        "realised_median_tau_mean": round(statistics.fmean(realised_taus), 4)
            if realised_taus else None,
        "realised_median_tau_median": round(statistics.median(realised_taus), 4)
            if realised_taus else None,
        "realised_pval_mean": round(statistics.fmean(realised_pvals), 4)
            if realised_pvals else None,
        "note": (
            "Monte-Carlo synthetic-ranking simulation. Each simulation "
            "fabricates a 4-arch (or 8-cell) ranking per candidate with "
            "pairwise Kendall tau-b centred on true_tau, then runs the "
            "same permutation procedure as the real test at a reduced "
            "perm budget. Estimate is approximate but informative."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _band(tau: float | None) -> str:
    if tau is None:
        return "unknown"
    if tau >= THRESHOLD:
        return "context-divergence-artefact (>=0.67)"
    if tau >= LOW_BAND:
        return "mixed-signal (>=0.33, <0.67)"
    return "interaction-real (<0.33)"


def _load_score_tables() -> dict[str, dict[str, Any]]:
    """Return the three views: NovelQA-only, QASPER-only, cross-dataset."""
    if not NOVELQA_GOLD.exists():
        raise SystemExit(f"missing NovelQA gold file: {NOVELQA_GOLD}")
    if not QASPER_GOLD.exists():
        raise SystemExit(f"missing QASPER gold file: {QASPER_GOLD}")
    nq = json.loads(NOVELQA_GOLD.read_text(encoding="utf-8"))
    qa = json.loads(QASPER_GOLD.read_text(encoding="utf-8"))
    nq_tbl: dict[str, dict[str, float]] = nq["per_candidate_per_arch_accuracy"]
    qa_tbl: dict[str, dict[str, float]] = qa["per_candidate_per_arch_answer_f1"]
    common = sorted(set(nq_tbl) & set(qa_tbl))
    nq_scores = {c: nq_tbl[c] for c in common}
    qa_scores = {c: qa_tbl[c] for c in common}

    # Cross-dataset 8-cell table.
    archs_cross = tuple([f"{a}__qasper" for a in ARCHS]
                        + [f"{a}__novelqa" for a in ARCHS])
    cross_scores: dict[str, dict[str, float]] = {}
    for cand in common:
        n_row = nq_scores[cand]
        q_row = qa_scores[cand]
        row: dict[str, float] = {}
        ok = True
        for arch in ARCHS:
            vn = n_row.get(arch)
            vq = q_row.get(arch)
            if not isinstance(vn, (int, float)) or not isinstance(vq, (int, float)):
                ok = False
                break
            row[f"{arch}__qasper"] = float(vq)
            row[f"{arch}__novelqa"] = float(vn)
        if ok:
            cross_scores[cand] = row

    return {
        "novelqa": {"scores": nq_scores, "archs": ARCHS,
                    "n_questions": nq.get("n_questions_in_slice")},
        "qasper": {"scores": qa_scores, "archs": ARCHS,
                   "n_questions": qa.get("calibration_pool_size")},
        "cross": {"scores": cross_scores, "archs": archs_cross,
                  "n_questions": (nq.get("n_questions_in_slice") or 0)
                                  + (qa.get("calibration_pool_size") or 0)},
    }


def _stats_for_view(name: str, view: dict[str, Any]) -> dict[str, Any]:
    scores = view["scores"]
    archs = view["archs"]
    rankings = _rankings_from(scores, archs)
    taus = _pairwise_taus(rankings)
    ci = bootstrap_median_ci(taus, n_boot=N_BOOT, seed=SEED)
    perm = permutation_pvalue(scores, archs, n_perm=N_PERM, seed=SEED)
    pwr = power_estimate(
        archs,
        n_candidates=len(rankings),
        true_tau=TRUE_TAU_FOR_POWER,
        n_sims=N_POWER_SIMS,
        n_perm=N_POWER_PERMS,
        alpha=ALPHA,
        seed=SEED,
    )
    return {
        "view": name,
        "n_questions_per_cell": view.get("n_questions"),
        "n_candidates": len(rankings),
        "n_pairs": len(taus),
        "observed_median_tau": ci["median_tau"],
        "observed_band": _band(ci["median_tau"]),
        "bootstrap_95ci": {"ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                           "n_boot": ci["n_boot"]},
        "permutation_test": perm,
        "power_analysis": pwr,
    }


def main() -> int:
    views = _load_score_tables()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "analysis_date": "2026-05-20",
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "alpha": ALPHA,
        "true_tau_for_power": TRUE_TAU_FOR_POWER,
        "rng_seed": SEED,
        "citation": "Dror et al. 2018, The Hitchhiker's Guide to Testing "
                    "Statistical Significance in NLP (\\cite{dror2018hitchhiker}).",
        "results": {},
    }
    for name, view in views.items():
        print(f"\n=== view={name} ===", flush=True)
        if not view["scores"]:
            summary["results"][name] = {"error": "no rankable candidates"}
            print("  (skipped - no rankable candidates)")
            continue
        block = _stats_for_view(name, view)
        summary["results"][name] = block
        print(f"  observed median tau-b = {block['observed_median_tau']} "
              f"({block['observed_band']})")
        print(f"  bootstrap 95% CI      = [{block['bootstrap_95ci']['ci_low']}, "
              f"{block['bootstrap_95ci']['ci_high']}]  (n_boot={N_BOOT})")
        print(f"  permutation p-value   = {block['permutation_test']['p_value_one_sided']} "
              f"(one-sided, n_perm={N_PERM})")
        print(f"  Monte-Carlo power     = {block['power_analysis']['power']} "
              f"(true_tau={TRUE_TAU_FOR_POWER}, n_sims={N_POWER_SIMS})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
