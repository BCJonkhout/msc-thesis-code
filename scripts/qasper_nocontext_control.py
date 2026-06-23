"""QASPER closed-book (no-context) memorization negative control.

The NovelQA novels are public-domain classics the answerer has likely seen in
pretraining, so a no-context control is needed to separate reading from recall.
QASPER papers are not memorized in the same way, but the only way to *show* that
rather than assert it is to run the same closed-book control on QASPER and report
the floor. Unlike NovelQA, QASPER gold is local, so Answer-F1 is scored here with
the official multi-reference token-F1 metric -- no Codabench round-trip.

Held identical to the flat full-context QASPER run (item set, answerer, T=0, and
scoring) except that the answerer is prompted closed-book, with no document:

  - Item set: ``load_qasper_full`` (calibration QIDs excluded, >=2 q/paper) ->
    the exact 955 questions over 249 papers the main study scored.
  - Prompt: a dedicated closed-book template that states the paper is unavailable
    and forces a parametric-knowledge best guess. Reusing the shared
    context-template with an empty context instead makes the answerer ask for the
    missing context rather than guessing, which measures refusal, not recall.
  - Answerer: ``gemini-3.1-flash-lite-preview`` at T=0, max_tokens 256 -- the
    main-study single answerer, same provider adapter.
  - Scoring: ``answer_f1_against_references`` against the local gold answers,
    identical to ``_score_item`` for QASPER.

``nocontext`` is the closed-book floor; ``flat - nocontext`` is the share of the
flat Answer-F1 that requires actually reading the paper. Idempotent + resumable
(rows keyed by ``(paper_id, question_id)`` are skipped on re-launch).

Output: outputs/main_study/qasper_nocontext_predictions.jsonl, and on completion
merges the QASPER floor into outputs/main_study/memorization_control.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pilot.cli.step_3_dry_run import load_qasper_full
from pilot.env import load_env
from pilot.eval.metrics import answer_f1_against_references
from pilot.providers.base import CacheControl
from pilot.providers.factory import get_provider

_ANSWERER_MODEL = "gemini-3.1-flash-lite-preview"
_ANSWERER_PROVIDER = "gemini"
_TEMPERATURE = 0.0
_MAX_TOKENS = 256

# Closed-book probe: the answerer has no document, so the shared context-template
# ("answer based on the provided context") makes it ask for the missing context
# instead of guessing. This dedicated prompt forces a parametric-knowledge best
# guess and keeps the answer format comparable to the with-document run, so the
# Answer-F1 comparison stays apples-to-apples.
_CLOSED_BOOK_PREFIX = (
    "You are answering a question about a specific research paper that you do not "
    "have in front of you.\n"
    "Answer from your own knowledge with your single best guess.\n"
    "Be concise and specific; for yes/no questions reply with one word.\n"
    "Do not ask for the paper and do not say you lack context. Always commit to a "
    "direct best-guess answer.\n\n"
    "Question: "
)

_DATA_ROOT = _REPO / "data"
_OUT_DIR = _REPO / "outputs" / "main_study"
_OUT_PATH = _OUT_DIR / "qasper_nocontext_predictions.jsonl"
_MEMO_PATH = _OUT_DIR / "memorization_control.json"
_FLAT_QASPER_F1 = 0.4573  # main-study flat full-context QASPER mean Answer-F1


def _load_done(path: Path) -> dict[tuple[str, str], float]:
    done: dict[tuple[str, str], float] = {}
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
                continue
            pid, qid = row.get("paper_id"), row.get("question_id")
            if pid and qid and isinstance(row.get("answer_f1"), (int, float)):
                done[(pid, qid)] = row["answer_f1"]
    return done


def _answer_one(provider, item: dict) -> dict:
    prompt = _CLOSED_BOOK_PREFIX + item["question"] + "\n\nAnswer:"
    result = provider.call(
        prompt,
        model=_ANSWERER_MODEL,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        cache_control=CacheControl.DISABLED,
    )
    af1 = answer_f1_against_references(result.text, item["gold_answers"])
    return {
        "dataset": "qasper",
        "paper_id": item["paper_id"],
        "question_id": item["question_id"],
        "predicted_answer": result.text,
        "answer_f1": af1,
    }


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, default=_OUT_PATH)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    items = load_qasper_full(_DATA_ROOT)
    done = _load_done(args.out)
    todo = [it for it in items if (it["paper_id"], it["question_id"]) not in done]
    n_papers = len({it["paper_id"] for it in items})
    print(f"[qasper-nocontext] {len(items)} questions / {n_papers} papers; "
          f"done {len(done)}; to do {len(todo)}", file=sys.stderr)

    if todo:
        provider = get_provider(_ANSWERER_PROVIDER)
        lock = threading.Lock()
        fh = args.out.open("a", encoding="utf-8")
        written = failures = 0

        def _flush(row: dict) -> None:
            nonlocal written
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                written += 1
                if written % 100 == 0:
                    print(f"[qasper-nocontext] {written}/{len(todo)}", file=sys.stderr)

        try:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futs = {pool.submit(_answer_one, provider, it): it for it in todo}
                for fut in as_completed(futs):
                    it = futs[fut]
                    try:
                        _flush(fut.result())
                    except Exception as exc:  # noqa: BLE001
                        failures += 1
                        print(f"[qasper-nocontext] FAIL {it['paper_id']}/"
                              f"{it['question_id']}: {exc!r}", file=sys.stderr)
        finally:
            fh.close()
        print(f"[qasper-nocontext] wrote {written}, {failures} failures", file=sys.stderr)

    # Final floor over the complete output.
    allrows = _load_done(args.out)
    floor = statistics.mean(allrows.values()) if allrows else 0.0
    print(f"\n[qasper-nocontext] closed-book Answer-F1 = {floor:.4f} "
          f"over {len(allrows)} questions; flat = {_FLAT_QASPER_F1:.4f}; "
          f"reading lift = {_FLAT_QASPER_F1 - floor:+.4f}")

    # Add a QASPER block alongside the existing (flat) NovelQA fields; do not
    # restructure them (tables_main_study.py reads the NovelQA keys directly).
    memo = json.loads(_MEMO_PATH.read_text(encoding="utf-8")) if _MEMO_PATH.exists() else {}
    memo["qasper"] = {
        "nocontext_answer_f1": round(floor, 4),
        "flat_answer_f1": _FLAT_QASPER_F1,
        "lift": round(_FLAT_QASPER_F1 - floor, 4),
        "n_questions": len(allrows),
    }
    _MEMO_PATH.write_text(json.dumps(memo, indent=2), encoding="utf-8")
    print(f"[qasper-nocontext] merged into {_MEMO_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
