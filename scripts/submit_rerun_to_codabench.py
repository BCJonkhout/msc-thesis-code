"""Submit Phase G RERUN (2026-05-19) RAPTOR + GraphRAG predictions to Codabench.

The 2026-05-16 Codabench-vs-consensus validation pass exposed a ~25pp
median gap between the consensus oracle and the official leaderboard
labels (GraphRAG ~30pp). The 2026-05-19 rerun re-generated RAPTOR +
GraphRAG predictions across the 9-candidate grid on the 15-question
3-novel slice (B41 excluded — deterministic BGE-M3 failure on specific
B41 chunks blocked the cache builds). Those new predictions have never
hit Codabench, so the rerun's consensus-oracle Kendall tau-b is not yet
gold-validated.

This driver submits all 18 (cand x arch) pairs for the rerun's RAPTOR
and GraphRAG predictions and recovers per-novel correctness strings
from ``scoring_stdout`` (the platform's MC-only scoring container
crashes on ``cr_gen_score``; the T/F strings are recoverable from the
log even when the submission lands in ``Failed``).

Submission strategy: three batches of 6, parallel within each batch
via ``ThreadPoolExecutor(max_workers=6)``, ~30s wait between batches.
The submission function is HTTP-bound so threads are appropriate.

Output:
  ``code/outputs/sanity/codabench_rerun_gold_20260519.json``

Each submission record carries the candidate slug, architecture, the
rerun run id, the Codabench submission id, the per-novel correctness
slice over the 15-question pool (B01 / B08 / B50 only), the
calibration-only accuracy under gold, and the scoring stdout excerpt.

The script writes after every batch so partial progress survives a
mid-batch crash.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from pilot.env import load_env  # noqa: E402
from pilot.codabench.format import write_submission_zip  # noqa: E402
from pilot.codabench.submit import submit_zip  # noqa: E402
from pilot.codabench.idempotency import (  # noqa: E402
    already_submitted,
    atomic_write_json,
    job_key,
    load_prior_records,
    split_results,
)
from pilot.codabench.extract_score import (  # noqa: E402
    fetch_correctness_strings,
    _calibration_indices,
    _norm_title,
)

LOCAL_SCORES_RERUN_PATH = (
    CODE_ROOT / "outputs" / "sanity" / "novelqa_local_scores_rerun_20260519.json"
)
DATA_ROOT = CODE_ROOT / "data"
ZIP_ROOT = CODE_ROOT / "outputs" / "codabench_zips" / "rerun_20260519"
OUT_JSON = CODE_ROOT / "outputs" / "sanity" / "codabench_rerun_gold_20260519.json"

ARCHS = ("raptor", "graphrag")
INCLUDED_NOVELS = ("B01", "B08", "B50")  # B41 excluded — matches the rerun pool
BATCH_SIZE = 6
INTER_BATCH_SLEEP_S = 30.0
RETRY_MAX = 3
RETRY_BACKOFF_S = 10.0


# ──────────────────────────────────────────────────────────────────────
# Per-job logic — runs inside the thread pool
# ──────────────────────────────────────────────────────────────────────

def _build_zip(pred_path: Path, zip_path: Path) -> dict[str, Any]:
    """Build the Codabench-shape submission zip. Idempotent."""
    return write_submission_zip(
        predictions_jsonl=pred_path,
        output_zip=zip_path,
        data_root=DATA_ROOT,
    )


def _submit_with_retry(zip_path: Path, name_hint: str) -> dict[str, Any]:
    """Submit a single zip with retries on transient errors.

    Returns a dict carrying submission_id + status, or a failure record.
    HTTP 429 is detected from the exception text and reported so the
    caller can fall back to a sequential drip.
    """
    last_exc: Exception | None = None
    rate_limited = False
    for attempt in range(1, RETRY_MAX + 1):
        try:
            result = submit_zip(zip_path=zip_path)
            return {
                "submission_id": result.submission_id,
                "status": result.status,
                "rate_limited": rate_limited,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            text = repr(exc).lower()
            if "429" in text or "rate" in text:
                rate_limited = True
            if attempt < RETRY_MAX:
                time.sleep(RETRY_BACKOFF_S * attempt)
    return {
        "submission_id": None,
        "status": "submit-failed",
        "rate_limited": rate_limited,
        "error": repr(last_exc) if last_exc else "unknown",
        "traceback": traceback.format_exc() if last_exc else None,
    }


def _calibration_acc_15(submission_id: int) -> dict[str, Any]:
    """Recover per-novel T/F from scoring_stdout, slice to the 15-question pool.

    Returns:
      {
        "per_novel": {B01: {title, correctness_slice, correct, total}, ...},
        "calibration_total_correct": int,
        "calibration_total_questions": int,
        "calibration_accuracy_n15": float,
        "overall_total_correct": int,
        "overall_total_questions": int,
        "scoring_stdout_excerpt": str,
      }
    The 15-question pool is the union of the 5 calibration QIDs in
    each of B01, B08, B50. B41's 5 calibration positions are
    intentionally excluded (matches the rerun grid's exclusion).
    """
    strings = fetch_correctness_strings(submission_id)
    if not strings:
        return {"error": "scoring_stdout not retrievable or empty"}

    bookmeta = json.loads((DATA_ROOT / "novelqa" / "bookmeta.json").read_text("utf-8"))
    cal_idx_by_novel = _calibration_indices(DATA_ROOT)

    cal_correct = cal_total = 0
    per_novel: dict[str, dict[str, Any]] = {}
    for novel_id in INCLUDED_NOVELS:
        idxs = cal_idx_by_novel.get(novel_id, [])
        title = bookmeta.get(novel_id, {}).get("title")
        if not title or not idxs:
            per_novel[novel_id] = {"title": title, "error": "no calibration indices / title"}
            continue
        s = strings.get(_norm_title(title), "")
        if not s:
            per_novel[novel_id] = {"title": title, "error": "no correctness string"}
            continue
        t = sum(1 for i in idxs if i < len(s) and s[i] == "T")
        n = sum(1 for i in idxs if i < len(s))
        cal_correct += t
        cal_total += n
        per_novel[novel_id] = {
            "title": title,
            "correctness_slice": "".join(s[i] if i < len(s) else "?" for i in idxs),
            "correct": t,
            "total": n,
        }

    total_t = sum(s.count("T") for s in strings.values())
    total_q = sum(len(s) for s in strings.values())

    # Build a compact excerpt of the per-novel T/F strings (one per line).
    excerpt_lines = [f'"{k}": "{v}"' for k, v in strings.items()]
    excerpt = "\n".join(excerpt_lines)

    return {
        "per_novel": per_novel,
        "calibration_total_correct": cal_correct,
        "calibration_total_questions": cal_total,
        "calibration_accuracy_n15": (
            round(cal_correct / cal_total, 4) if cal_total else None
        ),
        "overall_total_correct": total_t,
        "overall_total_questions": total_q,
        "overall_accuracy": round(total_t / total_q, 4) if total_q else None,
        "scoring_stdout_excerpt": excerpt,
    }


def _do_one(job: dict[str, Any]) -> dict[str, Any]:
    """Per-thread driver: format zip, submit, recover score, return record.

    All exceptions are caught — failures land in the record so the
    main thread can write them out.
    """
    candidate = job["candidate"]
    architecture = job["architecture"]
    rerun_run_id = job["rerun_run_id"]
    pred_path = job["pred_path"]
    zip_path = job["zip_path"]
    name_hint = job["name_hint"]

    record: dict[str, Any] = {
        "candidate": candidate,
        "architecture": architecture,
        "rerun_run_id": rerun_run_id,
        "predictions_path": str(pred_path),
        "zip_path": str(zip_path),
        "submission_name_hint": name_hint,
    }

    # Format
    try:
        stats = _build_zip(pred_path, zip_path)
        record["format_stats"] = {
            k: stats[k]
            for k in (
                "novels",
                "questions_total",
                "covered_by_predictions",
                "filled_with_placeholder",
            )
        }
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"format failed: {exc!r}"
        record["traceback"] = traceback.format_exc()
        return record

    # Submit + poll (this blocks until terminal status; that's fine —
    # we're inside a worker thread).
    submit_info = _submit_with_retry(zip_path, name_hint)
    record.update(submit_info)

    sub_id = submit_info.get("submission_id")
    if sub_id is None:
        return record

    # Tiny pause before reading scoring_stdout; the file is sometimes
    # still being uploaded when status flips terminal.
    time.sleep(2.0)
    try:
        calib = _calibration_acc_15(sub_id)
        record["scoring"] = calib
        record["codabench_accuracy_calibration_only_n15"] = calib.get(
            "calibration_accuracy_n15"
        )
        record["scoring_stdout_excerpt"] = calib.get("scoring_stdout_excerpt")
    except Exception as exc:  # noqa: BLE001
        record["scoring"] = {"error": f"score-extract failed: {exc!r}"}
        record["traceback"] = traceback.format_exc()
    return record


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def calibration_qids_n15() -> list[str]:
    pool = DATA_ROOT / "novelqa" / "calibration_pool.jsonl"
    out: list[str] = []
    with pool.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            if q["novel_id"] in INCLUDED_NOVELS:
                out.append(q["question_id"])
    return out


def build_jobs() -> list[dict[str, Any]]:
    data = json.loads(LOCAL_SCORES_RERUN_PATH.read_text("utf-8"))
    rerun_runids: dict[str, str] = data["rerun_runids"]
    jobs: list[dict[str, Any]] = []
    runs_root = CODE_ROOT / "outputs" / "runs"
    for candidate, run_id in rerun_runids.items():
        for arch in ARCHS:
            pred_path = runs_root / run_id / f"{arch}_predictions.jsonl"
            if not pred_path.exists():
                print(f"  WARN: missing predictions for {candidate}/{arch}: {pred_path}",
                      flush=True)
                continue
            zip_path = ZIP_ROOT / f"{candidate}__{arch}.zip"
            name_hint = f"phase_g_rerun_{candidate}_{arch}_20260519"
            jobs.append({
                "candidate": candidate,
                "architecture": arch,
                "rerun_run_id": run_id,
                "pred_path": pred_path,
                "zip_path": zip_path,
                "name_hint": name_hint,
            })
    return jobs


def _flush(payload: dict[str, Any]) -> None:
    atomic_write_json(OUT_JSON, payload)


def main() -> int:
    load_env()
    ZIP_ROOT.mkdir(parents=True, exist_ok=True)

    cal_qids = calibration_qids_n15()
    jobs = build_jobs()

    # Idempotent re-run. Load prior results and skip any job that already
    # carries a Codabench submission id: re-submitting would duplicate
    # against the throttled queue, and starting from an empty payload
    # would overwrite already-recovered gold scores. Failed jobs (no
    # submission id) are retried. New outcomes are merged into the
    # results dict keyed by (candidate, architecture).
    results: dict[tuple[str, str], dict[str, Any]] = load_prior_records(OUT_JSON)
    pending = [
        j for j in jobs
        if not already_submitted(results.get((j["candidate"], j["architecture"])))
    ]
    skipped = len(jobs) - len(pending)
    print(
        f"planned: {len(jobs)} | already submitted (skipped): {skipped} | "
        f"pending: {len(pending)}",
        flush=True,
    )
    for j in pending:
        print(f"  pending: {j['candidate']}/{j['architecture']} <- {j['rerun_run_id']}",
              flush=True)

    started_at = datetime.now(timezone.utc).isoformat()
    rate_limited_seen = False
    t_wall0 = time.monotonic()

    def _persist(extra: dict[str, Any] | None = None) -> None:
        subs, fails = split_results(results)
        payload: dict[str, Any] = {
            "submitted_at": started_at,
            "calibration_pool_qids_n15": cal_qids,
            "n_qids": len(cal_qids),
            "included_novels": list(INCLUDED_NOVELS),
            "batch_size": BATCH_SIZE,
            "inter_batch_sleep_s": INTER_BATCH_SLEEP_S,
            "submissions": subs,
            "failures": fails,
            "n_total": len(jobs),
            "n_skipped_already_submitted": skipped,
            "n_succeeded": len(subs),
            "n_failed": len(fails),
        }
        if extra:
            payload.update(extra)
        _flush(payload)

    _persist()

    # Batch the PENDING jobs only.
    batches: list[list[dict[str, Any]]] = [
        pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)
    ]
    for b_idx, batch in enumerate(batches, start=1):
        t_batch0 = time.monotonic()
        print(f"\n=== batch {b_idx}/{len(batches)} ({len(batch)} jobs) ===",
              flush=True)
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
            future_to_job = {pool.submit(_do_one, j): j for j in batch}
            for fut in as_completed(future_to_job):
                j = future_to_job[fut]
                try:
                    record = fut.result()
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "candidate": j["candidate"],
                        "architecture": j["architecture"],
                        "rerun_run_id": j["rerun_run_id"],
                        "error": f"worker exception: {exc!r}",
                        "traceback": traceback.format_exc(),
                    }
                results[job_key(record)] = record
                if record.get("submission_id") and not record.get("error"):
                    cb_acc = record.get("codabench_accuracy_calibration_only_n15")
                    print(
                        f"  ok: {j['candidate']:>34s}/{j['architecture']:<8s} "
                        f"sub_id={record['submission_id']} cal_acc={cb_acc}",
                        flush=True,
                    )
                else:
                    print(
                        f"  FAIL: {j['candidate']}/{j['architecture']}: "
                        f"{record.get('error', record.get('status'))}",
                        flush=True,
                    )
                if record.get("rate_limited"):
                    rate_limited_seen = True

        # Persist the merged results after each batch.
        _persist()
        print(f"  batch wallclock: {time.monotonic() - t_batch0:.1f}s", flush=True)

        if b_idx < len(batches):
            print(f"  sleeping {INTER_BATCH_SLEEP_S}s before next batch", flush=True)
            time.sleep(INTER_BATCH_SLEEP_S)

    total_wall = time.monotonic() - t_wall0
    _persist({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wallclock_seconds": round(total_wall, 1),
        "rate_limit_observed": rate_limited_seen,
    })
    subs, fails = split_results(results)
    print(
        f"\ndone: {len(subs)} ok / {len(fails)} failed / {len(jobs)} total "
        f"({skipped} skipped as already-submitted), wallclock {total_wall:.1f}s",
        flush=True,
    )
    print(f"wrote {OUT_JSON}", flush=True)
    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
