"""Per-(candidate, arch) accuracy on the 4-arch combined rerun grid.

Joins:
  Flat + Naive RAG predictions from outputs/rescore_20260516/<orig_run>/
  RAPTOR + GraphRAG predictions from outputs/rescore_20260519/<new_run>/

Aligns to the 15 calibration questions covered by all 4 architectures
(drop B41 questions — RAPTOR/GraphRAG could not be built for B41 in the
rerun because of a deterministic BGE-M3 failure on specific B41 chunks).
The Flat + Naive RAG accuracy reported here is therefore symmetric on
the same 15 questions, NOT the full 20-question pool used in the
2026-05-16 rescore.

Consensus oracle: per-question majority vote across the 9 candidates'
flat predictions on the 15-question slice (empty / unparsed cells
excluded from the vote).

Output: outputs/sanity/novelqa_local_scores_rerun_20260519.json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ARCHS = ("flat", "naive_rag", "raptor", "graphrag")
NEW_ARCHS = ("raptor", "graphrag")
OLD_ARCHS = ("flat", "naive_rag")

# Slice the 4-arch comparison to the 3 novels the rerun covers.
INCLUDED_NOVELS = ("B01", "B08", "B50")

# 9 candidates in the rerun (gemini-3.1-pro-preview is excluded in Phase G+rerun).
RERUN_CANDIDATES = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-latest",
    "grok-4-1-fast-non-reasoning",
    "grok-4-fast-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-0309-reasoning",
    "grok-4.3",
)

RUNID_PAT_NEW = re.compile(r"\[step3-dry-run\] run_id=([0-9a-f-]+)\s")


def _runid_for_rerun(log: Path) -> str | None:
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = RUNID_PAT_NEW.search(line)
            if m:
                return m.group(1)
    return None


def _runid_for_original(log: Path) -> str | None:
    """The original Phase G pareto log emits ``run_id=...`` once."""
    rids: list[str] = []
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "run_id=" not in line:
                continue
            for tok in line.split():
                if tok.startswith("run_id="):
                    rids.append(tok.split("=", 1)[1].rstrip(","))
    return rids[-1] if rids else None


def _question_options(calib_path: Path) -> dict[tuple[str, str], list[str]]:
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


def _modal(letters: list[str]) -> tuple[str | None, int, int, dict[str, int]]:
    if not letters:
        return None, 0, 0, {}
    c = Counter(letters)
    top = max(c.values())
    modal = sorted([k for k, v in c.items() if v == top])[0]
    return modal, top, len(letters), dict(c)


def _discover_original_runids(sanity_dir: Path) -> dict[str, str]:
    """Map candidate label -> ORIGINAL Phase G run_id (for Flat/Naive RAG)."""
    out: dict[str, str] = {}
    for log in sorted(sanity_dir.glob("pareto_novelqa_g_*.log")):
        cand = log.stem.replace("pareto_novelqa_g_novelqa-", "")
        rid = _runid_for_original(log)
        if rid is None:
            continue
        out[cand] = rid
    return out


def _discover_rerun_runids(logs_dir: Path) -> dict[str, str]:
    """Map candidate label -> RERUN run_id (for RAPTOR/GraphRAG)."""
    out: dict[str, str] = {}
    for log in sorted(logs_dir.glob("phase2par_*.log")):
        cand = log.stem.replace("phase2par_", "")
        rid = _runid_for_rerun(log)
        if rid is None:
            continue
        out[cand] = rid
    return out


def main() -> int:
    code_root = Path(__file__).resolve().parents[1]
    sanity = code_root / "outputs" / "sanity"
    calib = code_root / "data" / "novelqa" / "calibration_pool.jsonl"
    rescore_old = code_root / "outputs" / "rescore_20260516"
    rescore_new = code_root / "outputs" / "rescore_20260519"
    logs_rerun = code_root / "outputs" / "phase_g_rerun_2026-05-18" / "logs"

    options_by_q = _question_options(calib)
    all_questions = sorted(options_by_q.keys())
    # 15-question slice — drop B41
    questions = [(p, q) for (p, q) in all_questions if p in INCLUDED_NOVELS]
    assert len(questions) == 15, f"Expected 15 questions in 3-novel slice, got {len(questions)}"

    orig_runids = _discover_original_runids(sanity)
    rerun_runids = _discover_rerun_runids(logs_rerun)

    cand_labels: list[str] = list(RERUN_CANDIDATES)
    missing_orig = [c for c in cand_labels if c not in orig_runids]
    missing_rerun = [c for c in cand_labels if c not in rerun_runids]
    if missing_orig:
        raise SystemExit(f"missing original run_id mappings: {missing_orig}")
    if missing_rerun:
        raise SystemExit(f"missing rerun run_id mappings: {missing_rerun}")

    # Load all predictions: parsed[(cand, arch, paper, qid)] = (letter, empty)
    parsed: dict[tuple[str, str, str, str], tuple[str | None, bool]] = {}
    for cand in cand_labels:
        orig_rid = orig_runids[cand]
        new_rid = rerun_runids[cand]
        for arch in OLD_ARCHS:
            src = rescore_old / orig_rid / f"{arch}_predictions_rescored.jsonl"
            for (paper, qid), row in _load_rescored(src).items():
                if paper not in INCLUDED_NOVELS:
                    continue
                parsed[(cand, arch, paper, qid)] = (
                    row.get("predicted_letter_rescored"),
                    bool(row.get("empty_predicted_answer")),
                )
        for arch in NEW_ARCHS:
            src = rescore_new / new_rid / f"{arch}_predictions_rescored.jsonl"
            for (paper, qid), row in _load_rescored(src).items():
                parsed[(cand, arch, paper, qid)] = (
                    row.get("predicted_letter_rescored"),
                    bool(row.get("empty_predicted_answer")),
                )

    # Consensus oracle from flat predictions on the 15-question slice.
    consensus_full: dict[tuple[str, str], dict] = {}
    for paper, qid in questions:
        letters: list[str] = []
        voters: list[str] = []
        for cand in cand_labels:
            letter, empty = parsed.get((cand, "flat", paper, qid), (None, False))
            if letter and not empty:
                letters.append(letter)
                voters.append(cand)
        modal, top, total, dist = _modal(letters)
        if modal is None:
            continue
        consensus_full[(paper, qid)] = {
            "letter": modal,
            "support": top,
            "total_flat_votes": total,
            "voters": voters,
            "distribution": dist,
        }

    # Leave-one-out consensus per candidate.
    consensus_loo: dict[str, dict[tuple[str, str], dict]] = {}
    for excluded in cand_labels:
        c_loo: dict[tuple[str, str], dict] = {}
        for paper, qid in questions:
            letters: list[str] = []
            for cand in cand_labels:
                if cand == excluded:
                    continue
                letter, empty = parsed.get((cand, "flat", paper, qid), (None, False))
                if letter and not empty:
                    letters.append(letter)
            modal, top, total, dist = _modal(letters)
            if modal is None:
                continue
            c_loo[(paper, qid)] = {
                "letter": modal,
                "support": top,
                "total_flat_votes": total,
            }
        consensus_loo[excluded] = c_loo

    def _score(consensus_map, cand: str, arch: str) -> dict:
        n_questions = 0
        n_parsed = 0
        n_empty = 0
        n_unparsed_nonempty = 0
        n_correct = 0
        for paper, qid in questions:
            if (paper, qid) not in consensus_map:
                continue
            n_questions += 1
            letter, empty = parsed.get((cand, arch, paper, qid), (None, False))
            if empty:
                n_empty += 1
                continue
            if letter is None:
                n_unparsed_nonempty += 1
                continue
            n_parsed += 1
            if letter == consensus_map[(paper, qid)]["letter"]:
                n_correct += 1
        accuracy = round(n_correct / n_parsed, 4) if n_parsed else None
        return {
            "n_correct": n_correct,
            "n_parsed": n_parsed,
            "n_empty": n_empty,
            "n_unparsed_nonempty": n_unparsed_nonempty,
            "n_questions": n_questions,
            "accuracy_vs_consensus": accuracy,
        }

    scores_full: dict[tuple[str, str], dict] = {}
    scores_loo: dict[tuple[str, str], dict] = {}
    for cand in cand_labels:
        for arch in ARCHS:
            scores_full[(cand, arch)] = _score(consensus_full, cand, arch)
            scores_loo[(cand, arch)] = _score(consensus_loo[cand], cand, arch)

    strong = sum(1 for c in consensus_full.values() if c["support"] >= 7)
    moderate = sum(1 for c in consensus_full.values() if 4 <= c["support"] <= 6)
    weak = sum(1 for c in consensus_full.values() if c["support"] <= 3)

    out = {
        "schema_version": 1,
        "method": (
            "4-arch combined grid on 15-question 3-novel slice (B01, B08, "
            "B50). Flat + Naive RAG predictions joined from "
            "outputs/rescore_20260516/<orig_run>/; RAPTOR + GraphRAG joined "
            "from outputs/rescore_20260519/<rerun_run>/. B41 questions "
            "EXCLUDED (deterministic BGE-M3 failure on specific B41 chunks "
            "blocked RAPTOR + GraphRAG cache builds for B41). Flat / Naive "
            "RAG sliced to the same 15 questions to keep the comparison "
            "symmetric. Empty cells excluded from both consensus votes and "
            "per-candidate accuracy denominators."
        ),
        "rerun_date": "2026-05-19",
        "n_candidates": len(cand_labels),
        "included_novels": list(INCLUDED_NOVELS),
        "n_questions_in_slice": len(questions),
        "n_questions_in_consensus_full": len(consensus_full),
        "consensus_strength_full": {
            "strong_ge_7_of_9": strong,
            "moderate_4_to_6": moderate,
            "weak_le_3": weak,
        },
        "orig_runids": orig_runids,
        "rerun_runids": rerun_runids,
        "per_candidate_full": {
            cand: {arch: scores_full[(cand, arch)] for arch in ARCHS}
            for cand in cand_labels
        },
        "per_candidate_loo": {
            cand: {arch: scores_loo[(cand, arch)] for arch in ARCHS}
            for cand in cand_labels
        },
        "consensus_full_by_question": {
            f"{p}/{q}": v for (p, q), v in consensus_full.items()
        },
    }
    out_path = sanity / "novelqa_local_scores_rerun_20260519.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"NovelQA local-consensus accuracy (RERUN 2026-05-19, 4-arch, "
          f"n={len(consensus_full)} consensus questions, "
          f"{len(cand_labels)} candidates)")
    print(f"  consensus strength (full): strong>=7/9={strong}, "
          f"moderate 4-6={moderate}, weak<=3={weak}")
    print()
    for label, scores in (("FULL consensus", scores_full),
                          ("LOO consensus", scores_loo)):
        print(f"=== {label} ===")
        print(f"{'Candidate':<46} {'flat':>8} {'naive':>8} {'raptor':>8} "
              f"{'graphrag':>10} {'mean':>8}")
        print("-" * 100)
        rows = []
        for cand in cand_labels:
            accs = [scores[(cand, a)]["accuracy_vs_consensus"] for a in ARCHS]
            present = [a for a in accs if a is not None]
            mean = round(sum(present) / len(present), 4) if present else None
            rows.append((cand, accs, mean))
        rows.sort(key=lambda r: (r[2] or 0), reverse=True)
        for cand, accs, mean in rows:
            f = lambda v: f"{v:.3f}" if isinstance(v, float) else "  -  "
            print(f"{cand:<46} "
                  f"{f(accs[0]):>8} {f(accs[1]):>8} "
                  f"{f(accs[2]):>8} {f(accs[3]):>10} "
                  f"{f(mean):>8}")
        print()
    print(f"[novelqa_local_score_rerun_20260519] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
