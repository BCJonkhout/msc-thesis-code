"""NovelQA closed-book (no-context) memorization negative control.

Bounds how much of the flat full-context NovelQA result is the answerer
*reading* the novel versus *recalling* a memorized public-domain text.

The control answers each main-study NovelQA question from the question +
its multiple-choice options ALONE, with the document context held empty.
Everything else is held identical to the flat full-context run so the two
accuracies are directly comparable:

  - Item set: ``load_novelqa_full`` (B48 excluded, calibration QIDs
    excluded, min 2 questions/novel, full text on disk) -> the exact 1494
    questions over 60 public-domain novels the main study scored on
    Codabench.
  - Prompt: ``qa_multiplechoice_literature`` (no-abstention MC template),
    rendered through the same ``_render_prompt`` / ``_format_options_block``
    path as flat, but with ``context=""`` (question + options only).
  - Answerer: ``gemini-3.1-flash-lite-preview`` at temperature 0,
    max_tokens 256 -- the main-study single answerer (configs/models.yaml
    final_answerer.primary), same provider adapter (GeminiProvider).
  - Letter parsing: ``parse_mc_answer`` with the option-text list in
    A,B,C,... order, identical to ``_score_item`` in step_3_dry_run.

The only deliberate difference from flat is the empty context. The lift
``flat - nocontext`` is the share of flat's accuracy that requires the
novel text; ``nocontext`` itself is the closed-book / memorization floor.

Output row schema matches what ``pilot.codabench.format`` reads
(``dataset``, ``paper_id``/``novel_id``, ``question_id``,
``predicted_letter``), so the predictions JSONL feeds straight into
``write_submission_zip``.

Idempotent + resumable: rows already present in the output JSONL (keyed by
``(novel_id, question_id)``) are skipped, so the ~1494-call run can be
re-launched after an interruption without re-spending budget.

Usage::

    .venv/Scripts/python.exe scripts/novelqa_nocontext_control.py
    .venv/Scripts/python.exe scripts/novelqa_nocontext_control.py --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pilot.architectures.base import _render_prompt  # same MC rendering as flat
from pilot.cli.step_3_dry_run import load_novelqa_full  # same 1494-item set as flat
from pilot.env import load_env
from pilot.providers.base import CacheControl
from pilot.providers.factory import get_provider
from pilot.sanity.mc_postprocessor import parse_mc_answer  # same letter parser

_ANSWERER_MODEL = "gemini-3.1-flash-lite-preview"
_ANSWERER_PROVIDER = "gemini"
_PROMPT_STYLE = "literature"  # no-abstention MC template (qa_multiplechoice_literature)
_TEMPERATURE = 0.0
_MAX_TOKENS = 256  # matches run_flat default

_DATA_ROOT = _REPO / "data"
_OUT_DIR = _REPO / "outputs" / "main_study"
_OUT_PATH = _OUT_DIR / "novelqa_nocontext_predictions.jsonl"


def _load_done(path: Path) -> set[tuple[str, str]]:
    """(novel_id, question_id) pairs already written to the output JSONL."""
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard crash
            nid = row.get("novel_id") or row.get("paper_id")
            qid = row.get("question_id")
            if nid and qid:
                done.add((nid, qid))
    return done


def _answer_one(provider, item: dict) -> dict:
    """Render empty-context MC prompt, call answerer, parse letter -> row."""
    options = item["options"] or {}
    prompt = _render_prompt(
        context="",  # closed-book: no novel text, question + options only
        query=item["question"],
        options=options,
        prompt_style=_PROMPT_STYLE,
    )
    result = provider.call(
        prompt,
        model=_ANSWERER_MODEL,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        cache_control=CacheControl.DISABLED,
    )
    # Identical to _score_item: option text list in A,B,C,... order.
    opt_text_list = [options[k] for k in sorted(options.keys())]
    predicted_letter = parse_mc_answer(result.text, opt_text_list)
    return {
        "dataset": "novelqa",
        "paper_id": item["paper_id"],   # BID, what the format module reads
        "novel_id": item["paper_id"],   # explicit alias for clarity
        "question_id": item["question_id"],
        "predicted_answer": result.text,
        "predicted_letter": predicted_letter,
    }


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="parallel answerer calls (Gemini handles this; 1 = sequential).",
    )
    ap.add_argument("--out", type=Path, default=_OUT_PATH)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    items = load_novelqa_full(_DATA_ROOT)
    done = _load_done(args.out)
    todo = [it for it in items if (it["paper_id"], it["question_id"]) not in done]

    n_novels = len({it["paper_id"] for it in items})
    print(
        f"[nocontext] item set: {len(items)} questions / {n_novels} novels "
        f"(load_novelqa_full); already done: {len(done)}; to do: {len(todo)}",
        file=sys.stderr,
    )
    if not todo:
        print("[nocontext] nothing to do; output complete.", file=sys.stderr)
        return 0

    provider = get_provider(_ANSWERER_PROVIDER)

    write_lock = threading.Lock()
    out_fh = args.out.open("a", encoding="utf-8")
    written = 0
    failures = 0

    def _flush(row: dict) -> None:
        nonlocal written
        with write_lock:
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()
            import os as _os
            try:
                _os.fsync(out_fh.fileno())
            except (OSError, AttributeError):
                pass
            written += 1
            if written % 100 == 0:
                print(f"[nocontext] {written}/{len(todo)} answered", file=sys.stderr)

    try:
        if args.concurrency <= 1:
            for it in todo:
                try:
                    _flush(_answer_one(provider, it))
                except Exception as exc:  # noqa: BLE001 - record + continue
                    failures += 1
                    print(
                        f"[nocontext] FAIL {it['paper_id']}/{it['question_id']}: {exc!r}",
                        file=sys.stderr,
                    )
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futs = {pool.submit(_answer_one, provider, it): it for it in todo}
                for fut in as_completed(futs):
                    it = futs[fut]
                    try:
                        _flush(fut.result())
                    except Exception as exc:  # noqa: BLE001
                        failures += 1
                        print(
                            f"[nocontext] FAIL {it['paper_id']}/{it['question_id']}: {exc!r}",
                            file=sys.stderr,
                        )
    finally:
        out_fh.close()

    print(
        f"[nocontext] done: wrote {written} rows, {failures} failures. "
        f"Output: {args.out}",
        file=sys.stderr,
    )
    # Non-zero only if nothing got written this run (hard failure).
    return 0 if (written > 0 or failures == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
