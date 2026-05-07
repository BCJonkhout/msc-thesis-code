"""Format Step 3 predictions into a NovelQA Codabench submission zip.

NovelQA's leaderboard expects a zip containing
``res_mc/res_mc.json`` whose schema is::

    {
      "<novel title>": ["A", "B", "C", "D", ...],
      ...
    }

Each list is one entry per question for that novel, in the same QID
order as the source JSON in ``Data/PublicDomain/<BID>.json``.
Missing predictions (e.g. when only the calibration subset is
predicted) are filled with the placeholder letter "A". This is
methodologically the lowest-impact baseline — it does not improve
the architecture's score on unanswered questions, but it lets the
platform accept the submission so we get a real accuracy number on
the questions we DID predict, mixed into a known-baseline floor on
the rest.

Usage::

    from pilot.codabench import write_submission_zip
    write_submission_zip(
        predictions_jsonl=Path("outputs/runs/<run_id>/flat_predictions.jsonl"),
        output_zip=Path("outputs/codabench/flat_submission.zip"),
    )
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

# The placeholder letter used for questions that were not in the
# pilot's calibration pool. "A" is documented in the writeup as an
# explicit baseline; anything else (random, "I do not know") would
# require LLM-judge fallback we do not need.
NOVELQA_PLACEHOLDER_LETTER = "A"


def _project_data_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3] / "data"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _question_order_per_novel(data_root: Path) -> dict[str, list[str]]:
    """Per-novel list of QIDs in the order they appear in questions.jsonl.

    The JSONL was written by ``pilot.data.download.download_novelqa``
    by iterating ``Data/PublicDomain/B*.json`` and preserving each
    file's key insertion order, which is the canonical order the
    leaderboard expects.
    """
    questions = _load_jsonl(data_root / "novelqa" / "questions.jsonl")
    per_novel: dict[str, list[str]] = {}
    for q in questions:
        nid = q["novel_id"]
        per_novel.setdefault(nid, []).append(q["question_id"])
    return per_novel


def _novel_titles(data_root: Path) -> dict[str, str]:
    """Map novel_id (BID) to title, from bookmeta.json."""
    meta_path = data_root / "novelqa" / "bookmeta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {bid: rec["title"] for bid, rec in meta.items() if "title" in rec}


def _predictions_by_question(predictions_jsonl: Path) -> dict[tuple[str, str], str]:
    """Index a predictions JSONL by (novel_id, question_id) → predicted letter.

    Skips QASPER rows (which have no `predicted_letter`) and rows
    where the parser could not extract a letter. The per-row
    identifier field is ``paper_id`` regardless of dataset (set by
    ``step_3_dry_run.run_dry_run``); for NovelQA rows it carries
    the BID (e.g. "B01") so we re-name it ``novel_id`` here.
    """
    rows = _load_jsonl(predictions_jsonl)
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        if row.get("dataset") != "novelqa":
            continue
        letter = row.get("predicted_letter")
        if not isinstance(letter, str) or letter not in {"A", "B", "C", "D"}:
            continue
        novel_id = row.get("novel_id") or row.get("paper_id")
        question_id = row.get("question_id")
        if not novel_id or not question_id:
            continue
        out[(novel_id, question_id)] = letter.upper()
    return out


def build_res_mc(
    predictions_jsonl: Path,
    *,
    data_root: Path | None = None,
    placeholder: str = NOVELQA_PLACEHOLDER_LETTER,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Build res_mc.json content and a stats dict describing the fill rate.

    Returns:
        (res_mc, stats) where:
          - res_mc maps novel_title to a list of letters in QID order
          - stats has counts of {covered, missing, total} questions
    """
    data_root = data_root or _project_data_root()
    qid_order = _question_order_per_novel(data_root)
    titles = _novel_titles(data_root)
    preds = _predictions_by_question(predictions_jsonl)

    res_mc: dict[str, list[str]] = {}
    covered = 0
    missing = 0
    for novel_id, qids in qid_order.items():
        title = titles.get(novel_id)
        if not title:
            continue  # skip novels with no title metadata
        letters: list[str] = []
        for qid in qids:
            pred = preds.get((novel_id, qid))
            if pred:
                letters.append(pred)
                covered += 1
            else:
                letters.append(placeholder)
                missing += 1
        res_mc[title] = letters

    stats = {
        "novels": len(res_mc),
        "questions_total": covered + missing,
        "covered_by_predictions": covered,
        "filled_with_placeholder": missing,
        "placeholder_letter": placeholder,
        "predictions_jsonl": str(predictions_jsonl),
    }
    return res_mc, stats


def write_submission_zip(
    predictions_jsonl: Path,
    output_zip: Path,
    *,
    data_root: Path | None = None,
    placeholder: str = NOVELQA_PLACEHOLDER_LETTER,
) -> dict[str, Any]:
    """Write a Codabench-ready submission.zip containing res_mc/res_mc.json.

    Returns the stats dict from ``build_res_mc`` plus the output path.
    """
    res_mc, stats = build_res_mc(
        predictions_jsonl, data_root=data_root, placeholder=placeholder
    )
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("res_mc/res_mc.json", json.dumps(res_mc, ensure_ascii=False))
    stats["output_zip"] = str(output_zip)
    stats["output_zip_bytes"] = output_zip.stat().st_size
    return stats


def main() -> int:
    """CLI: build a submission zip from a predictions JSONL.

    Usage:
        python -m pilot.codabench.format \\
            --predictions outputs/runs/<run_id>/flat_predictions.jsonl \\
            --out outputs/codabench/flat_submission.zip
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--placeholder", default=NOVELQA_PLACEHOLDER_LETTER, choices=list("ABCD")
    )
    args = parser.parse_args()
    stats = write_submission_zip(
        predictions_jsonl=args.predictions,
        output_zip=args.out,
        placeholder=args.placeholder,
    )
    print(json.dumps(stats, indent=2))
    return 0 if stats["covered_by_predictions"] > 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
