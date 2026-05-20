"""Aggregate QASPER rerun (2026-05-20) gold Answer-F1 per (candidate, arch).

Walks the per-candidate run directories produced by
``code/outputs/qasper_rerun_2026-05-20/run_qasper_all.sh`` and writes
the canonical gold-scored summary at
``code/outputs/sanity/qasper_rerun_gold_20260520.json``.

Selection rule for the candidate -> run_id map: the most recent
``code/outputs/runs/<run_id>/`` whose ``step_3_dry_run_*.json``
verdict has ``datasets == ["qasper"]`` and matches the candidate's
``(answerer_provider, answerer_model)`` pair AND was started on or
after 2026-05-20T00:00:00Z (so we never pick up a stale May 13 run
that pre-dates the bug fixes).

Each per-arch JSONL row already carries the local QASPER ``answer_f1``
field (computed by ``step_3_dry_run.py``'s ``_score_item`` against
``gold_answers`` via the official multi-reference scorer
``pilot.eval.metrics.answer_f1_against_references``). We re-aggregate
here so the gold-validated table is reproducible from the on-disk
predictions alone, independent of whatever macro was written to the
verdict JSON.

Output schema mirrors the NovelQA gold file shape to keep downstream
Kendall scripts straight to wire up.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from pilot.eval.metrics import answer_f1_against_references  # noqa: E402

RERUN_DATE = "2026-05-20"
RUN_DATE_FLOOR = datetime(2026, 5, 20, 0, 0, 0, tzinfo=timezone.utc)
ARCHS = ("flat", "naive_rag", "raptor", "graphrag")

# 9-candidate slate matches the NovelQA gold 2026-05-19 file. Provider
# + model strings as they appear in the step_3_dry_run verdict JSON.
CANDIDATES: list[dict[str, str]] = [
    {"label": "gemini-3.1-flash-lite-preview",
     "provider": "google", "model": "gemini-3.1-flash-lite-preview"},
    {"label": "gemini-flash-latest",
     "provider": "google", "model": "gemini-flash-latest"},
    {"label": "deepseek-v4-flash",
     "provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
    {"label": "deepseek-v4-pro",
     "provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
    {"label": "grok-4-1-fast-non-reasoning",
     "provider": "xai", "model": "grok-4-1-fast-non-reasoning"},
    {"label": "grok-4-fast-reasoning",
     "provider": "xai", "model": "grok-4-fast-reasoning"},
    {"label": "grok-4.20-0309-non-reasoning",
     "provider": "xai", "model": "grok-4.20-0309-non-reasoning"},
    {"label": "grok-4.20-0309-reasoning",
     "provider": "xai", "model": "grok-4.20-0309-reasoning"},
    {"label": "grok-4.3",
     "provider": "xai", "model": "grok-4.3"},
]

SANITY_DIR = CODE_ROOT / "outputs" / "sanity"
RUNS_DIR = CODE_ROOT / "outputs" / "runs"
OUT_PATH = SANITY_DIR / f"qasper_rerun_gold_{RERUN_DATE.replace('-', '')}.json"


# ──────────────────────────────────────────────────────────────────────
# Verdict -> run_id resolution
# ──────────────────────────────────────────────────────────────────────

def _load_qasper_verdicts() -> list[dict[str, Any]]:
    """Return verdict dicts for QASPER step_3 runs on/after the date floor."""
    out: list[dict[str, Any]] = []
    for path in sorted(SANITY_DIR.glob("step_3_dry_run_*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if list(d.get("datasets") or []) != ["qasper"]:
            continue
        ts_str = d.get("timestamp_utc")
        if not ts_str:
            continue
        try:
            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ts < RUN_DATE_FLOOR:
            continue
        d["_verdict_path"] = str(path)
        d["_verdict_ts"] = ts.isoformat()
        out.append(d)
    return out


def _pick_run_for_candidate(
    candidate: dict[str, str], verdicts: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Most recent verdict whose answerer matches this candidate."""
    matches = [
        v for v in verdicts
        if v.get("answerer_provider") == candidate["provider"]
        and v.get("answerer_model") == candidate["model"]
    ]
    if not matches:
        return None
    return max(matches, key=lambda v: v["_verdict_ts"])


# ──────────────────────────────────────────────────────────────────────
# Gold scoring per (cand, arch)
# ──────────────────────────────────────────────────────────────────────

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
    """Reconstruct the gold reference list with the same rule as
    ``step_3_dry_run.load_qasper_calibration``."""
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


def _score_run_dir(
    run_id: str, pool: list[dict[str, Any]], dev_index: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per-arch gold F1 aggregation over the 20-question pool.

    Returns ``{arch: {n_questions, n_scored, n_correct_proxy,
    answer_f1_mean, per_question: [{paper_id, question_id, f1, predicted}]}}``.
    Missing rows (a candidate's run that errored on a cell) are
    counted as f1=0.0 so the macro stays comparable across candidates.
    """
    run_dir = RUNS_DIR / run_id
    out: dict[str, dict[str, Any]] = {}
    pool_keys = [(p["paper_id"], p["question_id"]) for p in pool]
    for arch in ARCHS:
        path = run_dir / f"{arch}_predictions.jsonl"
        if not path.exists():
            out[arch] = {
                "error": f"missing predictions file: {path.relative_to(CODE_ROOT)}",
                "n_questions": len(pool_keys),
                "n_scored": 0,
                "answer_f1_mean": None,
                "per_question": [],
            }
            continue
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
        for paper_id, qid in pool_keys:
            row = by_key.get((paper_id, qid))
            if row is None:
                per_q.append({
                    "paper_id": paper_id,
                    "question_id": qid,
                    "answer_f1": 0.0,
                    "predicted_answer": None,
                    "missing": True,
                })
                f1s.append(0.0)
                continue
            # Re-score against gold so the table is reproducible
            # from the JSONL alone.
            gold = _gold_answers_for(dev_index.get(paper_id, {}), qid)
            pred = row.get("predicted_answer", "") or ""
            f1 = answer_f1_against_references(pred, gold) if gold else None
            if f1 is None:
                # Fall back to the row's own answer_f1 field; if both
                # absent the question is unscoreable (no gold).
                f1_val = row.get("answer_f1")
                f1 = float(f1_val) if isinstance(f1_val, (int, float)) else 0.0
            per_q.append({
                "paper_id": paper_id,
                "question_id": qid,
                "answer_f1": round(float(f1), 6),
                "predicted_answer": pred,
                "missing": False,
            })
            f1s.append(float(f1))
        out[arch] = {
            "n_questions": len(pool_keys),
            "n_scored": sum(1 for q in per_q if not q["missing"]),
            "n_missing": sum(1 for q in per_q if q["missing"]),
            "answer_f1_mean": round(statistics.fmean(f1s), 6) if f1s else None,
            "answer_f1_median": round(statistics.median(f1s), 6) if f1s else None,
            "per_question": per_q,
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    pool = _load_calibration_pool()
    dev_index = _load_dev_index()
    verdicts = _load_qasper_verdicts()
    print(f"loaded {len(pool)} calibration questions, "
          f"{len(verdicts)} QASPER verdicts on/after {RUN_DATE_FLOOR.date()}",
          flush=True)

    per_candidate: dict[str, dict[str, Any]] = {}
    pretty_table: dict[str, dict[str, float | None]] = {}
    run_ids: dict[str, str | None] = {}

    for cand in CANDIDATES:
        verdict = _pick_run_for_candidate(cand, verdicts)
        if verdict is None:
            print(f"  MISS {cand['label']}: no qualifying verdict", flush=True)
            per_candidate[cand["label"]] = {
                "answerer_provider": cand["provider"],
                "answerer_model": cand["model"],
                "run_id": None,
                "verdict_path": None,
                "per_arch": {a: {"error": "no run found"} for a in ARCHS},
            }
            pretty_table[cand["label"]] = {a: None for a in ARCHS}
            run_ids[cand["label"]] = None
            continue
        run_id = verdict["run_id"]
        per_arch = _score_run_dir(run_id, pool, dev_index)
        per_candidate[cand["label"]] = {
            "answerer_provider": cand["provider"],
            "answerer_model": cand["model"],
            "run_id": run_id,
            "verdict_path": verdict["_verdict_path"],
            "summary_provider": verdict.get("summary_provider"),
            "summary_model": verdict.get("summary_model"),
            "per_arch": per_arch,
        }
        pretty_table[cand["label"]] = {
            a: per_arch[a].get("answer_f1_mean") for a in ARCHS
        }
        run_ids[cand["label"]] = run_id
        f1s_str = " ".join(
            f"{a}={(per_arch[a].get('answer_f1_mean') or 0):.3f}"
            for a in ARCHS
        )
        print(f"  OK   {cand['label']:>34s} run={run_id} {f1s_str}", flush=True)

    # Per-architecture macro
    per_arch_means: dict[str, list[float]] = defaultdict(list)
    for cand_label, blk in per_candidate.items():
        for arch in ARCHS:
            v = blk.get("per_arch", {}).get(arch, {}).get("answer_f1_mean")
            if isinstance(v, (int, float)):
                per_arch_means[arch].append(v)

    per_arch_mean = {
        a: round(statistics.fmean(vs), 6) if vs else None
        for a, vs in per_arch_means.items()
    }
    per_arch_median = {
        a: round(statistics.median(vs), 6) if vs else None
        for a, vs in per_arch_means.items()
    }
    arch_ranking_by_mean = sorted(
        ARCHS, key=lambda a: (per_arch_mean.get(a) or -1), reverse=True
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "rerun_date": RERUN_DATE,
        "dataset": "qasper",
        "calibration_pool_size": len(pool),
        "n_candidates": len(CANDIDATES),
        "architectures": list(ARCHS),
        "summary_model_pin": {
            "provider": "google",
            "model": "gemini-3.1-flash-lite-preview",
            "rationale": "Shared cache key across answerer families; "
                         "matches the NovelQA rerun pin.",
        },
        "scorer": "pilot.eval.metrics.answer_f1_against_references "
                  "(multi-reference QASPER token-F1)",
        "run_ids": run_ids,
        "per_candidate_per_arch_answer_f1": pretty_table,
        "per_candidate_full": per_candidate,
        "per_architecture_mean_answer_f1": per_arch_mean,
        "per_architecture_median_answer_f1": per_arch_median,
        "architecture_ranking_by_mean": list(arch_ranking_by_mean),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Console table
    print("\nQASPER gold Answer-F1 per (candidate, architecture):")
    print(f"  {'candidate':<34s} {'flat':>7s} {'naive':>7s} {'raptor':>8s} {'graphrag':>9s}")
    for cand_label in pretty_table:
        row = pretty_table[cand_label]
        cells = [
            f"{row[a]:.4f}" if isinstance(row[a], (int, float)) else "  -   "
            for a in ARCHS
        ]
        print(f"  {cand_label:<34s} {cells[0]:>7s} {cells[1]:>7s} "
              f"{cells[2]:>8s} {cells[3]:>9s}")
    print(f"\nper-arch mean   : {per_arch_mean}")
    print(f"per-arch median : {per_arch_median}")
    print(f"arch ranking    : {arch_ranking_by_mean}")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
