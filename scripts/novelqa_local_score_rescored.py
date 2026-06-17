"""Score the re-parsed Phase G NovelQA predictions against a consensus oracle.

Inputs:
  outputs/rescore_20260516/<run_id>/<arch>_predictions_rescored.jsonl

Outputs:
  outputs/sanity/novelqa_local_scores_rescored.json

This is a re-implementation of ``novelqa_local_score.py`` that:

  1. Consumes the rescored prediction files (the parser fix is already
     applied to ``predicted_letter_rescored``).
  2. Treats ``empty_predicted_answer == True`` cells as MISSING data —
     they do NOT count against accuracy and do NOT contribute to the
     consensus vote. Under the old parser DeepSeek-pro's empty cells
     were being scored as wrong; that conflated empty-output (a
     provider headroom bug) with wrong-answer (a model-quality
     finding).
  3. Computes two consensus variants:

       * ``consensus_full``: modal flat-letter across ALL 9
         full-grid candidates (matches the original script).
       * ``consensus_loo``: leave-one-out consensus — for each
         candidate the consensus is recomputed from the OTHER 8
         candidates only, eliminating the self-reference bias
         where a candidate's own flat prediction biases its own
         flat-accuracy upward.

Run from ``code/``::

    .venv/Scripts/python.exe scripts/novelqa_local_score_rescored.py

Provenance: PILOT-ERA. The rescored variant of the consensus-oracle
NovelQA scoring (patched-parser predictions, empty cells excluded);
SUPERSEDED by held-out Codabench gold (``novelqa_codabench_accuracy.py``),
which the main study uses instead. Kept as a reproducibility record; not
the source of a current paper claim.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ARCHS = ("flat", "naive_rag", "raptor", "graphrag")


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


def _latest_run_id(log: Path) -> str | None:
    rids: list[str] = []
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "run_id=" not in line:
                continue
            rid = line.split("run_id=", 1)[1].split()[0].rstrip(",")
            rids.append(rid)
    return rids[-1] if rids else None


def _load_rescored(run_dir: Path, arch: str) -> dict[tuple[str, str], dict]:
    path = run_dir / f"{arch}_predictions_rescored.jsonl"
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


def _modal(letters: list[str]) -> tuple[str | None, int, int, dict[str, int]]:
    if not letters:
        return None, 0, 0, {}
    c = Counter(letters)
    top = max(c.values())
    modal = sorted([k for k, v in c.items() if v == top])[0]
    return modal, top, len(letters), dict(c)


def main() -> int:
    code_root = Path(__file__).resolve().parents[1]
    sanity = code_root / "outputs" / "sanity"
    runs = code_root / "outputs" / "runs"
    calib = code_root / "data" / "novelqa" / "calibration_pool.jsonl"
    rescore_root = code_root / "outputs" / "rescore_20260516"

    options_by_q = _question_options(calib)
    questions = sorted(options_by_q.keys())

    # Discover candidates from the same logs the original script uses.
    candidates: list[tuple[str, str, Path]] = []
    for log in sorted(sanity.glob("pareto_novelqa_g_*.log")):
        cand = log.stem.replace("pareto_novelqa_", "")
        rid = _latest_run_id(log)
        if rid is None:
            continue
        rd = rescore_root / rid
        # Only include candidates whose rescored dir actually exists.
        if rd.exists():
            candidates.append((cand, rid, rd))

    # parsed[(cand, arch, paper, qid)] = (letter_or_None, empty_flag)
    parsed: dict[tuple[str, str, str, str], tuple[str | None, bool]] = {}
    rows_by_key: dict[tuple[str, str, str, str], dict] = {}
    for cand, _, rd in candidates:
        for arch in ARCHS:
            preds = _load_rescored(rd, arch)
            for (paper, qid), row in preds.items():
                letter = row.get("predicted_letter_rescored")
                empty = bool(row.get("empty_predicted_answer"))
                parsed[(cand, arch, paper, qid)] = (letter, empty)
                rows_by_key[(cand, arch, paper, qid)] = row

    cand_labels = [c for c, _, _ in candidates]

    # ── Full consensus from flat ───────────────────────────────────
    consensus_full: dict[tuple[str, str], dict] = {}
    for paper, qid in questions:
        flat_letters: list[str] = []
        flat_voters: list[str] = []
        for cand in cand_labels:
            letter, empty = parsed.get((cand, "flat", paper, qid), (None, False))
            # Exclude empty AND unparsed cells from consensus vote.
            if letter and not empty:
                flat_letters.append(letter)
                flat_voters.append(cand)
        modal, top, total, dist = _modal(flat_letters)
        if modal is None:
            continue
        consensus_full[(paper, qid)] = {
            "letter": modal,
            "support": top,
            "total_flat_votes": total,
            "voters": flat_voters,
            "distribution": dist,
        }

    # ── Leave-one-out consensus per candidate (from flat) ─────────
    # consensus_loo[cand][(paper, qid)] = {...}
    consensus_loo: dict[str, dict[tuple[str, str], dict]] = {}
    for excluded in cand_labels:
        c_loo: dict[tuple[str, str], dict] = {}
        for paper, qid in questions:
            flat_letters: list[str] = []
            for cand in cand_labels:
                if cand == excluded:
                    continue
                letter, empty = parsed.get((cand, "flat", paper, qid), (None, False))
                if letter and not empty:
                    flat_letters.append(letter)
            modal, top, total, dist = _modal(flat_letters)
            if modal is None:
                continue
            c_loo[(paper, qid)] = {
                "letter": modal,
                "support": top,
                "total_flat_votes": total,
            }
        consensus_loo[excluded] = c_loo

    # ── Score every (cand, arch) under full and LOO ─────────────
    def _score(consensus_map: dict[tuple[str, str], dict],
               cand: str, arch: str) -> dict:
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

    # ── Consensus strength diagnostic ─────────────────────────────
    strong = sum(1 for c in consensus_full.values() if c["support"] >= 7)
    moderate = sum(1 for c in consensus_full.values() if 4 <= c["support"] <= 6)
    weak = sum(1 for c in consensus_full.values() if c["support"] <= 3)

    out = {
        "schema_version": 1,
        "method": (
            "cross-candidate consensus oracle from flat-full-context "
            "predictions, re-parsed with patched MC parser (b9ce51d). "
            "Empty predicted_answer cells are excluded from both consensus "
            "votes and per-candidate accuracy."
        ),
        "n_candidates": len(candidates),
        "n_questions_in_consensus_full": len(consensus_full),
        "consensus_strength_full": {
            "strong_ge_7_of_9": strong,
            "moderate_4_to_6": moderate,
            "weak_le_3": weak,
        },
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
    out_path = sanity / "novelqa_local_scores_rescored.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Text table — print both full and LOO
    print(f"NovelQA local-consensus accuracy (RESCORED, "
          f"n={len(consensus_full)} consensus questions, "
          f"{len(candidates)} candidates)")
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
    print(f"[novelqa_local_score_rescored] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
