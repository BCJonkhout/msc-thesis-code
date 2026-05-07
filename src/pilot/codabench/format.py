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
    leaderboard expects. Copyright-protected novels are read from the
    same HuggingFace snapshot's ``Data/CopyrightProtected/B*.json``
    files so the submission can include all 89 novels (the leaderboard
    expects every novel; predictions for novels we can't run end-to-end
    are filled with the placeholder letter).
    """
    questions = _load_jsonl(data_root / "novelqa" / "questions.jsonl")
    per_novel: dict[str, list[str]] = {}
    for q in questions:
        nid = q["novel_id"]
        per_novel.setdefault(nid, []).append(q["question_id"])

    # Also include the copyright-protected novels' QIDs from the HF snapshot.
    # Their questions live in the same NovelQA.zip but were not extracted to
    # questions.jsonl by download_novelqa (we skip them because the texts
    # are withheld). For the submission we still need the QID order so the
    # platform-side scorer can match indices.
    cp_path = _find_copyright_protected_qids()
    for nid, qids in cp_path.items():
        per_novel.setdefault(nid, []).extend(qids)

    return per_novel


def _find_copyright_protected_qids() -> dict[str, list[str]]:
    """Walk the HF NovelQA snapshot's CopyrightProtected/B*.json files
    and return per-BID question-id orderings. Returns {} if the
    snapshot isn't on disk."""
    import os
    import zipfile

    hf_cache = Path(
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface")
    )
    snapshots = hf_cache / "hub" / "datasets--NovelQA--NovelQA" / "snapshots"
    if not snapshots.exists():
        return {}
    for snap in snapshots.iterdir():
        zip_path = snap / "NovelQA.zip"
        if not zip_path.exists():
            continue
        out: dict[str, list[str]] = {}
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if not member.startswith("Data/CopyrightProtected/"):
                    continue
                if not member.endswith(".json"):
                    continue
                novel_id = Path(member).stem
                with zf.open(member) as fh:
                    qmap = json.loads(fh.read().decode("utf-8"))
                out[novel_id] = list(qmap.keys())
        return out
    return {}


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
    include_gen_stub: bool = True,
) -> dict[str, Any]:
    """Write a Codabench-ready submission.zip containing res_mc/res_mc.json.

    When ``include_gen_stub=True`` (default) the zip also contains
    ``res_gen/res_gen.json`` populated with the same per-novel
    structure (each value being a list of empty strings, one per
    question) plus a stub ``res_gen/key`` with a placeholder string.
    This works around a bug in the platform's scoring script that
    references ``cr_gen_score`` even when only res_mc was uploaded:

        UnboundLocalError: local variable 'cr_gen_score'
        referenced before assignment

    The stub generative answers are empty strings, which the
    GPT-4 judge will score uniformly wrong; the stub OpenAI key is
    not a real key and its calls will fail. The point is only to
    make the platform's scoring container reach the
    ``cr_gen_score`` print without crashing on the MC results.

    Returns the stats dict from ``build_res_mc`` plus the output
    path and a flag indicating whether the gen stub was included.
    """
    res_mc, stats = build_res_mc(
        predictions_jsonl, data_root=data_root, placeholder=placeholder
    )
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("res_mc/res_mc.json", json.dumps(res_mc, ensure_ascii=False))
        if include_gen_stub:
            res_gen = {title: ["" for _ in letters] for title, letters in res_mc.items()}
            zf.writestr("res_gen/res_gen.json", json.dumps(res_gen, ensure_ascii=False))
            # Empty key file; OpenAI calls in the platform's gen scorer
            # will fail but the scoring script will still set the variable
            # the platform downstream code references.
            zf.writestr("res_gen/key", "sk-stub-not-a-real-key")
    stats["output_zip"] = str(output_zip)
    stats["output_zip_bytes"] = output_zip.stat().st_size
    stats["include_gen_stub"] = include_gen_stub
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
    parser.add_argument(
        "--no-gen-stub",
        action="store_true",
        help=(
            "Skip the res_gen/ stub. Required only if you have a real "
            "GPT-4 key and want to submit to the generative subtask. "
            "Without this flag, an empty res_gen/res_gen.json + stub "
            "res_gen/key are added to work around a platform-side "
            "scoring-script crash on res_mc-only submissions."
        ),
    )
    args = parser.parse_args()
    stats = write_submission_zip(
        predictions_jsonl=args.predictions,
        output_zip=args.out,
        placeholder=args.placeholder,
        include_gen_stub=not args.no_gen_stub,
    )
    print(json.dumps(stats, indent=2))
    return 0 if stats["covered_by_predictions"] > 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
