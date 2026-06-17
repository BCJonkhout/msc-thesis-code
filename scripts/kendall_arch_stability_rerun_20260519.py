"""Kendall tau-b architecture rank stability — 2026-05-19 rerun grid.

Reads ``outputs/sanity/novelqa_local_scores_rerun_20260519.json`` and
emits ``outputs/sanity/novelqa_kendall_rerun_20260519.json``.

Reports three blocks for each consensus variant (full, LOO):

  1. ``__4arch``  — full 4-arch (flat, naive_rag, raptor, graphrag).
     This is the primary outcome of the rerun.
  2. ``__flat_naive`` — Flat + Naive RAG only sanity check; should
     approximately match the 2026-05-16 rescore result (50% stable
     pairs, median tau == 0.000) since these slices share the same
     underlying predictions, only the question denominator has shrunk
     from 20 to 15 questions.
  3. ``__jackknife_4arch`` — per-question jackknife over the 15
     calibration questions on the full 4-arch grid. For each question
     we drop it from every candidate's accuracy denominators, recompute
     the per-(cand, arch) accuracy, and recompute the median tau and
     stable-pair fraction. Reports min/median/max of jackknife medians
     and the count of questions whose individual removal would flip the
     primary decision-rule verdict (cross 0.33 or 0.67).

Decision rule (pre-registered Phase F.1 / phase_g_rerun_preregistration):
  tau >= 0.67 AND >=75% pairs STABLE => "context-divergence artefact"
  tau in [0.33, 0.67)                => "mixed signal"
  tau < 0.33                          => "interaction is real"

Provenance: PILOT-ERA. The 2026-05-19 rerun stage of the
architecture-rank-stability tau-b series (consensus-oracle -> rescored ->
held-out-gold lineage); reads the rerun consensus scores and adds the
per-question jackknife. Superseded by the held-out-gold variant. Kept as
a reproducibility record; not the source of a current paper claim.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

from pilot.cli.phase_f_kendall import _kendall_tau, _rank_by_f1


THRESHOLD = 0.67
LOW_BAND = 0.33

ARCHS_FULL = ("flat", "naive_rag", "raptor", "graphrag")
ARCHS_CLEAN = ("flat", "naive_rag")
INCLUDED_NOVELS = ("B01", "B08", "B50")

RUNID_PAT_NEW = re.compile(r"\[step3-dry-run\] run_id=([0-9a-f-]+)\s")


def _runid_for_rerun(log: Path) -> str | None:
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = RUNID_PAT_NEW.search(line)
            if m:
                return m.group(1)
    return None


def _runid_for_original(log: Path) -> str | None:
    rids: list[str] = []
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "run_id=" not in line:
                continue
            for tok in line.split():
                if tok.startswith("run_id="):
                    rids.append(tok.split("=", 1)[1].rstrip(","))
    return rids[-1] if rids else None


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


def _band(tau: float) -> str:
    if tau is None:
        return "unknown"
    if tau >= THRESHOLD:
        return "context-divergence-artefact (>=0.67)"
    if tau >= LOW_BAND:
        return "mixed-signal (>=0.33, <0.67)"
    return "interaction-real (<0.33)"


# ---------- Jackknife helpers ----------

NEW_ARCHS = ("raptor", "graphrag")
OLD_ARCHS = ("flat", "naive_rag")


def _load_rescored(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != "novelqa":
                continue
            key = (row["paper_id"], row["question_id"])
            out[key] = row
    return out


def _modal(letters: list[str]) -> str | None:
    if not letters:
        return None
    c = Counter(letters)
    top = max(c.values())
    return sorted([k for k, v in c.items() if v == top])[0]


def _jackknife(local_scores_meta: dict, code_root: Path) -> dict:
    """Per-question jackknife on the 4-arch full-consensus tau-b."""
    sanity = code_root / "outputs" / "sanity"
    rescore_old = code_root / "outputs" / "rescore_20260516"
    rescore_new = code_root / "outputs" / "rescore_20260519"
    calib = code_root / "data" / "novelqa" / "calibration_pool.jsonl"

    # Re-derive parsed lookup from the rescore files (mirrors local_score
    # script). This avoids storing the heavy per-question table inside the
    # local-score JSON output.
    all_questions: list[tuple[str, str]] = []
    with calib.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            all_questions.append((q["novel_id"], q["question_id"]))
    questions = [(p, q) for (p, q) in all_questions if p in INCLUDED_NOVELS]

    orig_runids = local_scores_meta["orig_runids"]
    rerun_runids = local_scores_meta["rerun_runids"]
    cand_labels = list(rerun_runids.keys())

    parsed: dict[tuple[str, str, str, str], tuple[str | None, bool]] = {}
    for cand in cand_labels:
        for arch in OLD_ARCHS:
            src = rescore_old / orig_runids[cand] / f"{arch}_predictions_rescored.jsonl"
            for (paper, qid), row in _load_rescored(src).items():
                if paper not in INCLUDED_NOVELS:
                    continue
                parsed[(cand, arch, paper, qid)] = (
                    row.get("predicted_letter_rescored"),
                    bool(row.get("empty_predicted_answer")),
                )
        for arch in NEW_ARCHS:
            src = rescore_new / rerun_runids[cand] / f"{arch}_predictions_rescored.jsonl"
            for (paper, qid), row in _load_rescored(src).items():
                parsed[(cand, arch, paper, qid)] = (
                    row.get("predicted_letter_rescored"),
                    bool(row.get("empty_predicted_answer")),
                )

    def _consensus_excluding(excluded_question: tuple[str, str] | None) -> dict:
        cons: dict[tuple[str, str], str] = {}
        for (paper, qid) in questions:
            if (paper, qid) == excluded_question:
                continue
            letters = []
            for cand in cand_labels:
                letter, empty = parsed.get((cand, "flat", paper, qid), (None, False))
                if letter and not empty:
                    letters.append(letter)
            modal = _modal(letters)
            if modal is not None:
                cons[(paper, qid)] = modal
        return cons

    def _per_cand_scores(cons: dict[tuple[str, str], str],
                         excluded_question: tuple[str, str] | None) -> dict:
        result: dict[str, dict[str, dict]] = {}
        for cand in cand_labels:
            arch_block: dict[str, dict] = {}
            for arch in ARCHS_FULL:
                n_parsed = 0
                n_correct = 0
                for (paper, qid) in questions:
                    if (paper, qid) == excluded_question:
                        continue
                    if (paper, qid) not in cons:
                        continue
                    letter, empty = parsed.get((cand, arch, paper, qid), (None, False))
                    if empty or letter is None:
                        continue
                    n_parsed += 1
                    if letter == cons[(paper, qid)]:
                        n_correct += 1
                acc = round(n_correct / n_parsed, 6) if n_parsed else None
                arch_block[arch] = {"accuracy_vs_consensus": acc}
            result[cand] = arch_block
        return result

    # Baseline (no question dropped) for cross-check
    base_cons = _consensus_excluding(None)
    base_scores = _per_cand_scores(base_cons, None)
    base_block = _kendall_block(base_scores, ARCHS_FULL, "jackknife baseline (no drop)")

    jk_records: list[dict] = []
    for (paper, qid) in questions:
        cons = _consensus_excluding((paper, qid))
        per_cand = _per_cand_scores(cons, (paper, qid))
        block = _kendall_block(per_cand, ARCHS_FULL, f"jk drop {paper}/{qid}")
        jk_records.append({
            "dropped_question": f"{paper}/{qid}",
            "n_candidates_ranked": block["n_candidates"],
            "median_tau": block["median_tau"],
            "mean_tau": block["mean_tau"],
            "stable_fraction": block["stable_fraction"],
            "n_stable": block["n_stable"],
            "n_pairs": block["n_pairs"],
            "band": _band(block["median_tau"]),
        })

    medians = [r["median_tau"] for r in jk_records if r["median_tau"] is not None]
    stable_pcts = [r["stable_fraction"] for r in jk_records if r["stable_fraction"] is not None]
    base_band = _band(base_block["median_tau"])
    n_flip = sum(1 for r in jk_records if r["band"] != base_band)

    return {
        "baseline": {
            "median_tau": base_block["median_tau"],
            "mean_tau": base_block["mean_tau"],
            "stable_fraction": base_block["stable_fraction"],
            "n_stable": base_block["n_stable"],
            "n_pairs": base_block["n_pairs"],
            "band": base_band,
        },
        "jackknife_summary": {
            "n_questions_dropped_one_at_a_time": len(jk_records),
            "median_tau_min": round(min(medians), 4) if medians else None,
            "median_tau_median": round(statistics.median(medians), 4) if medians else None,
            "median_tau_max": round(max(medians), 4) if medians else None,
            "median_tau_mean": round(statistics.fmean(medians), 4) if medians else None,
            "stable_fraction_min": round(min(stable_pcts), 4) if stable_pcts else None,
            "stable_fraction_max": round(max(stable_pcts), 4) if stable_pcts else None,
            "n_drops_that_flip_baseline_band": n_flip,
            "baseline_band": base_band,
        },
        "per_dropped_question": jk_records,
    }


def main() -> int:
    code_root = Path(__file__).resolve().parents[1]
    in_path = code_root / "outputs" / "sanity" / "novelqa_local_scores_rerun_20260519.json"
    out_path = code_root / "outputs" / "sanity" / "novelqa_kendall_rerun_20260519.json"
    data = json.loads(in_path.read_text(encoding="utf-8"))

    blocks: dict[str, dict] = {}
    for variant in ("full", "loo"):
        per_cand = data[f"per_candidate_{variant}"]
        blocks[f"{variant}__4arch"] = _kendall_block(
            per_cand, ARCHS_FULL, f"{variant} consensus, 4-arch"
        )
        blocks[f"{variant}__flat_naive"] = _kendall_block(
            per_cand, ARCHS_CLEAN, f"{variant} consensus, flat+naive_rag only"
        )

    primary = blocks["full__4arch"]
    primary_band = _band(primary["median_tau"])

    jackknife = _jackknife(data, code_root)

    out = {
        "schema_version": 1,
        "threshold": THRESHOLD,
        "low_band": LOW_BAND,
        "rerun_date": "2026-05-19",
        "primary_decision_band": primary_band,
        "primary_median_tau": primary["median_tau"],
        "primary_stable_fraction": primary["stable_fraction"],
        "n_candidates": primary["n_candidates"],
        "blocks": blocks,
        "jackknife_full_4arch": jackknife,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Kendall tau-b — RERUN 2026-05-19 (4-arch combined grid, 3-novel slice)")
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
    print()
    print(f"PRIMARY BAND: {primary_band}  "
          f"(median tau = {primary['median_tau']}, "
          f"stable fraction = {primary['stable_fraction']})")
    print()
    jk = jackknife["jackknife_summary"]
    print(f"Jackknife (n={jk['n_questions_dropped_one_at_a_time']} drops):")
    print(f"  median tau range: min={jk['median_tau_min']}, "
          f"median={jk['median_tau_median']}, max={jk['median_tau_max']}")
    print(f"  stable% range:    min={jk['stable_fraction_min']}, "
          f"max={jk['stable_fraction_max']}")
    print(f"  drops that flip baseline band: {jk['n_drops_that_flip_baseline_band']}")
    print(f"\n[kendall_arch_stability_rerun_20260519] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
