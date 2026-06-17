"""Re-score gemini-flash-latest QASPER predictions from the
2026-05-20 re-run (run_id ``20260520-a2ab12d4``).

The QASPER cell-set for ``gemini-flash-latest`` was rebuilt on top of
the ``gemini_provider.py`` thinking-model heuristic fix (now recognises
``-latest`` aliases as Gemini 2.5 thinking models with 4x max_tokens
headroom). The previous truncated predictions
(median visible output ~10 tokens) are superseded by this re-run's 80
predictions across the four architectures.

This script re-aggregates the gold Answer-F1 per (architecture, question)
against the QASPER dev-set gold reference set, using the official
QASPER multi-reference scorer
``pilot.eval.metrics.answer_f1_against_references``.

Output: ``code/outputs/sanity/qasper_gemini_flash_latest_rescored_20260520.json``

This file is deterministic and re-runnable: given the same predictions
and the same dev.jsonl gold table, the output is byte-identical.

Provenance: PILOT-ERA QASPER rescore for a single candidate
(gemini-flash-latest) after the thinking-model token-headroom fix; a
repair feeding the pilot QASPER rerun gold aggregation. The canonical
main-study QASPER scoring is on the ``build_scored_cells.py`` path. Kept
as a reproducibility record; not the source of a current paper claim.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from pilot.eval.metrics import answer_f1_against_references  # noqa: E402

RERUN_RUN_ID = "20260520-a2ab12d4"
CANDIDATE_LABEL = "gemini-flash-latest"
ARCHS = ("flat", "naive_rag", "raptor", "graphrag")

RUN_DIR = CODE_ROOT / "outputs" / "runs" / RERUN_RUN_ID
SANITY_DIR = CODE_ROOT / "outputs" / "sanity"
OUT_PATH = SANITY_DIR / "qasper_gemini_flash_latest_rescored_20260520.json"


def _load_calibration_pool() -> list[dict[str, Any]]:
    pool_path = CODE_ROOT / "data" / "qasper" / "calibration_pool.jsonl"
    out: list[dict[str, Any]] = []
    with pool_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_dev_index() -> dict[str, dict[str, Any]]:
    """paper_id -> paper dict from dev.jsonl (carries gold answers)."""
    dev_path = CODE_ROOT / "data" / "qasper" / "dev.jsonl"
    out: dict[str, dict[str, Any]] = {}
    with dev_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            p = json.loads(line)
            out[p["paper_id"]] = p
    return out


def _gold_answers_for(paper: dict[str, Any], question_id: str) -> list[str]:
    """Reconstruct gold references with the same rule as
    ``step_3_dry_run.load_qasper_calibration`` and
    ``collect_qasper_gold_20260520._gold_answers_for``."""
    qa = next(
        (q for q in paper.get("qas", []) if q.get("question_id") == question_id),
        None,
    )
    if qa is None:
        return []
    out: list[str] = []
    for ans in qa.get("answers", []) or []:
        a = ans.get("answer", {})
        if not isinstance(a, dict):
            continue
        ff = a.get("free_form_answer") or ""
        if ff and ff.strip():
            out.append(ff.strip())
        elif a.get("extractive_spans"):
            out.append(" ".join(s for s in a["extractive_spans"] if s))
        elif a.get("yes_no") is True:
            out.append("Yes")
        elif a.get("yes_no") is False:
            out.append("No")
        elif a.get("unanswerable"):
            out.append("")
    return out


def _score_arch(arch: str, pool: list[dict[str, Any]],
                dev_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = RUN_DIR / f"{arch}_predictions.jsonl"
    if not path.exists():
        raise SystemExit(f"missing predictions file: {path}")

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != "qasper":
                continue
            key = (row.get("paper_id", ""), row.get("question_id", ""))
            by_key[key] = row

    per_q: list[dict[str, Any]] = []
    f1s: list[float] = []
    for p in pool:
        pid, qid = p["paper_id"], p["question_id"]
        row = by_key.get((pid, qid))
        if row is None:
            per_q.append({"qid": qid, "paper_id": pid, "f1": 0.0, "missing": True})
            f1s.append(0.0)
            continue
        gold = _gold_answers_for(dev_index.get(pid, {}), qid)
        pred = row.get("predicted_answer", "") or ""
        if gold:
            f1 = answer_f1_against_references(pred, gold)
        else:
            f1_val = row.get("answer_f1")
            f1 = float(f1_val) if isinstance(f1_val, (int, float)) else 0.0
        per_q.append({
            "qid": qid,
            "paper_id": pid,
            "f1": round(float(f1), 6),
            "predicted_answer": pred,
            "missing": False,
        })
        f1s.append(float(f1))

    return {
        "n_questions": len(pool),
        "n_scored": sum(1 for q in per_q if not q["missing"]),
        "n_missing": sum(1 for q in per_q if q["missing"]),
        "answer_f1_mean": round(statistics.fmean(f1s), 6),
        "answer_f1_median": round(statistics.median(f1s), 6),
        "per_question": per_q,
    }


def main() -> int:
    pool = _load_calibration_pool()
    dev_index = _load_dev_index()

    per_arch: dict[str, dict[str, Any]] = {}
    per_arch_means: dict[str, float] = {}
    per_cell_f1: dict[str, list[dict[str, Any]]] = {}

    for arch in ARCHS:
        block = _score_arch(arch, pool, dev_index)
        per_arch[arch] = block
        per_arch_means[arch] = block["answer_f1_mean"]
        per_cell_f1[arch] = [
            {"qid": q["qid"], "f1": q["f1"]} for q in block["per_question"]
        ]
        print(f"  {arch:>10s}  mean={block['answer_f1_mean']:.6f}  "
              f"median={block['answer_f1_median']:.6f}  "
              f"n_scored={block['n_scored']}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "candidate": CANDIDATE_LABEL,
        "rerun_run_id": RERUN_RUN_ID,
        "scorer": "pilot.eval.metrics.answer_f1_against_references "
                  "(multi-reference QASPER token-F1)",
        "fix_reference": (
            "gemini_provider.py thinking-model heuristic extended to "
            "recognise -latest aliases (gemini-flash-latest, "
            "gemini-pro-latest) as Gemini 2.5 thinking models so the "
            "QASPER short-answer cap (256) is bumped 4x for headroom."
        ),
        "per_arch_means": per_arch_means,
        "per_cell_f1": per_cell_f1,
        "per_arch_full": per_arch,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
