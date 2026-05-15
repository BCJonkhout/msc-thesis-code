"""Score NovelQA predictions against a cross-candidate consensus oracle.

NovelQA's gold answers are leaderboard-held (Codabench). For the pilot
we approximate accuracy locally by treating the **majority-vote answer
across all candidates' flat-full-context predictions as the consensus
oracle** for each question. Every candidate then has its predictions
scored against that consensus.

Rationale and caveats:

- Flat full-context is the architecture with the most context evidence
  (the full 60–90k-token novel in the prompt); its predictions are
  the strongest local proxy for the right MC letter when we cannot
  consult the held-out gold.
- Cross-candidate majority vote across 9 distinct answerers (Google,
  xAI, OpenRouter) is more robust than picking any single model as
  oracle — disagreements get diluted, broad agreement reinforces a
  high-confidence consensus letter.
- This is a **pilot-grade** measurement. The main study will need
  actual Codabench-held gold answers because (a) the consensus is
  uncertain wherever the candidates split evenly, and (b) any
  systematic bias shared by all candidates (e.g. position bias on
  MC letters) propagates into the consensus undetected.

Output:

  outputs/sanity/novelqa_local_scores.json   structured scores
  stdout                                     text table

Run from ``code/``::

    .venv/Scripts/python.exe scripts/novelqa_local_score.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from pilot.sanity.mc_postprocessor import parse_mc_answer


def _latest_run_id(log: Path) -> str | None:
    rids: list[str] = []
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "run_id=" not in line:
                continue
            m = line.split("run_id=", 1)[1].split()[0].rstrip(",")
            rids.append(m)
    return rids[-1] if rids else None


def _load_predictions(run_dir: Path, arch: str) -> dict[tuple[str, str], dict]:
    """Map (paper_id, question_id) -> row for one arch's JSONL."""
    path = run_dir / f"{arch}_predictions.jsonl"
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
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


def _question_options(calib_path: Path) -> dict[tuple[str, str], list[str]]:
    """Read the calibration pool to recover each question's option-text
    list (needed by parse_mc_answer for the 'matches the option text'
    branch)."""
    out: dict[tuple[str, str], list[str]] = {}
    with calib_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            opts = q.get("Options") or {}
            opt_texts = [opts[k] for k in sorted(opts.keys())]
            out[(q["novel_id"], q["question_id"])] = opt_texts
    return out


def main() -> int:
    sanity = Path("outputs/sanity")
    runs = Path("outputs/runs")
    calib = Path("data/novelqa/calibration_pool.jsonl")
    options_by_q = _question_options(calib)

    # Discover all candidate run_ids from the NovelQA lane logs.
    candidates: list[tuple[str, Path]] = []
    for log in sorted(sanity.glob("pareto_novelqa_g_*.log")):
        cand = log.stem.replace("pareto_novelqa_", "")
        rid = _latest_run_id(log)
        if rid is None:
            continue
        rd = runs / rid
        if rd.exists():
            candidates.append((cand, rd))

    # ── Step 1: Parse every prediction into a letter ───────────────
    # parsed_letter[(cand, arch, paper, qid)] = "A" | "B" | None
    parsed: dict[tuple[str, str, str, str], str | None] = {}
    for cand, rd in candidates:
        for arch in ("flat", "naive_rag", "raptor", "graphrag"):
            preds = _load_predictions(rd, arch)
            for (paper, qid), row in preds.items():
                opts = options_by_q.get((paper, qid)) or []
                pred_text = row.get("predicted_answer") or ""
                # If the row already carries predicted_letter, trust it;
                # otherwise re-parse.
                letter = row.get("predicted_letter")
                if not letter:
                    letter = parse_mc_answer(pred_text, opts)
                parsed[(cand, arch, paper, qid)] = letter

    # ── Step 2: Build consensus oracle from flat predictions ───────
    # For each question, take the modal letter across all candidates'
    # flat predictions. Ties are broken by alphabetical order on
    # letter (deterministic). Questions where all candidates returned
    # None are skipped.
    consensus: dict[tuple[str, str], dict] = {}
    questions = sorted(options_by_q.keys())
    for (paper, qid) in questions:
        flat_letters: list[str] = []
        for cand, _ in candidates:
            letter = parsed.get((cand, "flat", paper, qid))
            if letter:
                flat_letters.append(letter)
        if not flat_letters:
            continue
        c = Counter(flat_letters)
        top_count = max(c.values())
        # Deterministic tie-break: alphabetical
        modal = sorted([k for k, v in c.items() if v == top_count])[0]
        consensus[(paper, qid)] = {
            "letter": modal,
            "support": top_count,
            "total_flat_votes": len(flat_letters),
            "distribution": dict(c),
        }

    # ── Step 3: Score every (candidate, arch) against consensus ────
    scores: dict[tuple[str, str], dict] = {}
    for cand, _ in candidates:
        for arch in ("flat", "naive_rag", "raptor", "graphrag"):
            n_total = 0
            n_correct = 0
            n_unparsed = 0
            n_questions_with_consensus = 0
            for (paper, qid) in questions:
                if (paper, qid) not in consensus:
                    continue
                n_questions_with_consensus += 1
                letter = parsed.get((cand, arch, paper, qid))
                if letter is None:
                    n_unparsed += 1
                    continue
                n_total += 1
                if letter == consensus[(paper, qid)]["letter"]:
                    n_correct += 1
            accuracy = round(n_correct / n_total, 4) if n_total else None
            scores[(cand, arch)] = {
                "n_correct": n_correct,
                "n_parsed": n_total,
                "n_unparsed": n_unparsed,
                "n_questions": n_questions_with_consensus,
                "accuracy_vs_consensus": accuracy,
            }

    # ── Step 4: Consensus quality diagnostic ──────────────────────
    # How many questions had STRONG consensus (&gt;= 7 of 9 candidates
    # agreed on the same letter on flat) vs WEAK consensus.
    strong = sum(1 for c in consensus.values() if c["support"] >= 7)
    moderate = sum(1 for c in consensus.values() if 4 <= c["support"] <= 6)
    weak = sum(1 for c in consensus.values() if c["support"] <= 3)

    # ── Step 5: Dump JSON + print table ────────────────────────────
    out = {
        "schema_version": 1,
        "method": "cross-candidate consensus oracle from flat-full-context predictions",
        "n_candidates": len(candidates),
        "n_questions_in_consensus": len(consensus),
        "consensus_strength": {
            "strong_ge_7_of_9": strong,
            "moderate_4_to_6": moderate,
            "weak_le_3": weak,
        },
        "per_candidate": {
            cand: {
                arch: scores[(cand, arch)]
                for arch in ("flat", "naive_rag", "raptor", "graphrag")
            }
            for cand, _ in candidates
        },
        "consensus_by_question": {
            f"{p}/{q}": v for (p, q), v in consensus.items()
        },
    }
    out_path = sanity / "novelqa_local_scores.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Text table
    print(f"NovelQA local-consensus accuracy (n={len(consensus)} questions, "
          f"{len(candidates)} candidates contributing to consensus)")
    print(f"  consensus strength: strong (&gt;=7/9)={strong}, "
          f"moderate (4-6)={moderate}, weak (&lt;=3)={weak}")
    print()
    print(f"{'Candidate':<46} {'flat':>8} {'naive':>8} {'raptor':>8} {'graphrag':>10} {'mean':>8}")
    print("-" * 100)
    rows = []
    for cand, _ in candidates:
        accs = [scores[(cand, a)]["accuracy_vs_consensus"] for a in
                ("flat", "naive_rag", "raptor", "graphrag")]
        present = [a for a in accs if a is not None]
        mean = round(sum(present) / len(present), 4) if present else None
        rows.append((cand, accs, mean))
    rows.sort(key=lambda r: (r[2] or 0), reverse=True)
    for cand, accs, mean in rows:
        f = lambda v: f"{v:.3f}" if isinstance(v, float) else "  -  "
        print(
            f"{cand:<46} "
            f"{f(accs[0]):>8} {f(accs[1]):>8} "
            f"{f(accs[2]):>8} {f(accs[3]):>10} "
            f"{f(mean):>8}"
        )
    print()
    print(f"[novelqa_local_score] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
