"""One-off driver: submit all re-scored Phase G predictions to Codabench.

Iterates the 10 candidate runs × 4 architectures under
``code/outputs/rescore_20260516/``, builds a Codabench submission
zip for each, submits it via ``pilot.codabench.submit``, recovers
the per-novel correctness strings from ``scoring_stdout`` (because
the platform's official accept path crashes on MC-only submissions),
slices to the 20-question calibration pool, and writes a comparison
JSON against the local consensus-oracle scores.

Throttle: 60s between submission starts to be friendly to the
Codabench shared queue. Per-submission retry budget is 3 for
transient network / 5xx failures.

Provenance (see docs/CODEMAP.md): PILOT-ERA Codabench submission driver. This is
the one-off that produced the 2026-05-16 consensus-oracle-vs-gold comparison over
the rescored Phase G grid (the ~25pp median gap that motivated scoring NovelQA
against the official leaderboard). It is not part of the canonical main-study
pipeline; the main study uses novelqa_codabench_accuracy / build_scored_cells.
Kept for the historical audit trail of the gold-vs-consensus validation.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from pilot.env import load_env  # noqa: E402
from pilot.codabench.format import write_submission_zip  # noqa: E402
from pilot.codabench.submit import submit_zip  # noqa: E402
from pilot.codabench.extract_score import (  # noqa: E402
    fetch_correctness_strings,
    _calibration_indices,
    _norm_title,
)

RESCORE_ROOT = CODE_ROOT / "outputs" / "rescore_20260516"
ZIP_ROOT = CODE_ROOT / "outputs" / "codabench_zips" / "rescore_20260516"
DATA_ROOT = CODE_ROOT / "data"
LOCAL_SCORES_PATH = CODE_ROOT / "outputs" / "sanity" / "novelqa_local_scores_rescored.json"
OUT_JSON = CODE_ROOT / "outputs" / "sanity" / "codabench_vs_consensus_2026-05-16.json"
OUT_MD = CODE_ROOT.parent / "thesis-msc" / "notes" / "codabench_validation_2026-05-16.md"

ARCHS = ["flat", "naive_rag", "raptor", "graphrag"]
SUBMISSION_THROTTLE_S = 60.0
RETRY_MAX = 3


def build_run_to_candidate_map() -> dict[str, str]:
    """Map timestamp run_id -> 'g_novelqa-<model-slug>' by inspecting ledger."""
    runs_root = CODE_ROOT / "outputs" / "runs"
    # Hard-coded mapping from generate-stage model name to the
    # candidate slug used in local consensus scores. The slug matches
    # the per_candidate_full keys in novelqa_local_scores_rescored.json.
    model_to_slug = {
        "grok-4.20-0309-reasoning": "g_novelqa-grok-4.20-0309-reasoning",
        "deepseek/deepseek-v4-flash": "g_novelqa-deepseek-v4-flash",
        "gemini-flash-latest": "g_novelqa-gemini-flash-latest",
        "grok-4-fast-reasoning": "g_novelqa-grok-4-fast-reasoning",
        "gemini-3.1-pro-preview": "g_novelqa-gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro": "g_novelqa-deepseek-v4-pro",
        "gemini-3.1-flash-lite-preview": "g_novelqa-gemini-3.1-flash-lite-preview",
        "grok-4-1-fast-non-reasoning": "g_novelqa-grok-4-1-fast-non-reasoning",
        "grok-4.20-0309-non-reasoning": "g_novelqa-grok-4.20-0309-non-reasoning",
        "grok-4.3": "g_novelqa-grok-4.3",
    }
    mapping: dict[str, str] = {}
    for d in sorted(RESCORE_ROOT.iterdir()):
        if not d.is_dir():
            continue
        run_id = d.name
        ledger = runs_root / run_id / "ledger.jsonl"
        if not ledger.exists():
            mapping[run_id] = None
            continue
        models: set[str] = set()
        with ledger.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("stage") == "generate":
                    m = rec.get("model")
                    if m:
                        models.add(m)
        if len(models) != 1:
            print(f"  warn: run {run_id} has multiple generate models: {models}", flush=True)
        model_name = next(iter(models)) if models else None
        mapping[run_id] = model_to_slug.get(model_name)
    return mapping


def short_candidate(slug: str) -> str:
    """Strip the 'g_novelqa-' prefix for compact filenames / table cells."""
    return slug.removeprefix("g_novelqa-") if slug else "unknown"


def calibration_qids() -> list[str]:
    pool = DATA_ROOT / "novelqa" / "calibration_pool.jsonl"
    out: list[str] = []
    with pool.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            out.append(json.loads(line)["question_id"])
    return out


def compute_calibration_accuracy(submission_id: int) -> dict:
    """Fetch scoring_stdout, slice to calibration positions, compute accuracy.

    Returns:
        {
          "correctness_per_novel": { "B01": "TTTFT", ... },
          "calibration_total_correct": int,
          "calibration_total_questions": int,
          "calibration_accuracy": float,
          "overall_total_correct": int,
          "overall_total_questions": int,
        }
    """
    strings = fetch_correctness_strings(submission_id)
    if not strings:
        return {"error": "scoring_stdout not retrievable or empty"}

    bookmeta = json.loads(
        (DATA_ROOT / "novelqa" / "bookmeta.json").read_text(encoding="utf-8")
    )
    cal_idx_by_novel = _calibration_indices(DATA_ROOT)

    cal_correct = cal_total = 0
    cal_per_novel: dict[str, dict] = {}
    for novel_id, idxs in cal_idx_by_novel.items():
        title = bookmeta.get(novel_id, {}).get("title")
        if not title:
            continue
        s = strings.get(_norm_title(title), "")
        if not s:
            cal_per_novel[novel_id] = {"title": title, "error": "no correctness string"}
            continue
        t = sum(1 for i in idxs if i < len(s) and s[i] == "T")
        n = sum(1 for i in idxs if i < len(s))
        cal_correct += t
        cal_total += n
        cal_per_novel[novel_id] = {
            "title": title,
            "correctness_slice": "".join(s[i] if i < len(s) else "?" for i in idxs),
            "correct": t,
            "total": n,
        }

    total_t = sum(s.count("T") for s in strings.values())
    total_q = sum(len(s) for s in strings.values())

    return {
        "calibration_per_novel": cal_per_novel,
        "calibration_total_correct": cal_correct,
        "calibration_total_questions": cal_total,
        "calibration_accuracy": round(cal_correct / cal_total, 4) if cal_total else None,
        "overall_total_correct": total_t,
        "overall_total_questions": total_q,
        "overall_accuracy": round(total_t / total_q, 4) if total_q else None,
    }


def load_local_consensus_scores() -> dict:
    return json.loads(LOCAL_SCORES_PATH.read_text(encoding="utf-8"))


def submit_one(zip_path: Path, name_hint: str) -> dict:
    """Submit a single zip with retries on transient errors.

    Returns a dict with submission_id, status, and any error info.
    Does NOT raise — callers should inspect the dict.
    """
    last_exc: Exception | None = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            print(f"    submit attempt {attempt}/{RETRY_MAX} for {name_hint}", flush=True)
            result = submit_zip(zip_path=zip_path)
            return {
                "submission_id": result.submission_id,
                "status": result.status,
                "raw_status": result.status,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"    submit attempt {attempt} failed: {exc!r}", flush=True)
            if attempt < RETRY_MAX:
                time.sleep(30 * attempt)  # back off
    return {
        "submission_id": None,
        "status": "submit-failed",
        "error": repr(last_exc) if last_exc else "unknown",
        "traceback": traceback.format_exc() if last_exc else None,
    }


def main() -> int:
    load_env()
    ZIP_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    run_map = build_run_to_candidate_map()
    print(f"discovered {len(run_map)} run dirs", flush=True)
    for run_id, slug in run_map.items():
        print(f"  {run_id} -> {slug}", flush=True)

    cal_qids = calibration_qids()
    local_scores = load_local_consensus_scores()
    per_candidate_local = local_scores.get("per_candidate_full", {})

    started = datetime.now(timezone.utc).isoformat()
    submissions: list[dict] = []

    work = []
    for run_id, slug in run_map.items():
        if not slug:
            continue
        for arch in ARCHS:
            pred_path = RESCORE_ROOT / run_id / f"{arch}_predictions_rescored.jsonl"
            if not pred_path.exists():
                continue
            work.append((run_id, slug, arch, pred_path))

    print(f"\ntotal submissions planned: {len(work)}", flush=True)

    for i, (run_id, slug, arch, pred_path) in enumerate(work, start=1):
        short = short_candidate(slug)
        zip_name = f"{short}__{arch}.zip"
        zip_path = ZIP_ROOT / zip_name
        name_hint = f"phase_g_rescore_{short}_{arch}_2026-05-16"
        print(f"\n[{i}/{len(work)}] {short} / {arch}", flush=True)

        record: dict = {
            "candidate": slug,
            "candidate_short": short,
            "architecture": arch,
            "run_id": run_id,
            "predictions_path": str(pred_path),
            "zip_path": str(zip_path),
            "submission_name_hint": name_hint,
        }

        # Build zip (idempotent — overwrites prior content).
        try:
            stats = write_submission_zip(
                predictions_jsonl=pred_path,
                output_zip=zip_path,
                data_root=DATA_ROOT,
            )
            record["format_stats"] = {
                k: stats[k]
                for k in (
                    "novels",
                    "questions_total",
                    "covered_by_predictions",
                    "filled_with_placeholder",
                )
            }
            print(
                f"  zip built: {stats['covered_by_predictions']} real / "
                f"{stats['questions_total']} total",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"format failed: {exc!r}"
            submissions.append(record)
            print(f"  FORMAT FAILED: {exc!r}", flush=True)
            continue

        # Throttle between submissions (skip on the first).
        if i > 1:
            print(f"  throttle: sleeping {SUBMISSION_THROTTLE_S}s", flush=True)
            time.sleep(SUBMISSION_THROTTLE_S)

        submit_info = submit_one(zip_path, name_hint)
        record.update(submit_info)

        sub_id = submit_info.get("submission_id")
        if sub_id is None:
            submissions.append(record)
            continue

        # Allow scoring container a moment after terminal status before
        # fetching the log — sometimes the file is still being uploaded.
        time.sleep(5.0)

        try:
            calib = compute_calibration_accuracy(sub_id)
            record["scoring"] = calib
            cb_acc = calib.get("calibration_accuracy")
            print(
                f"  calibration: {calib.get('calibration_total_correct')}/"
                f"{calib.get('calibration_total_questions')} = {cb_acc}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            record["scoring"] = {"error": f"score-extract failed: {exc!r}"}
            print(f"  SCORE EXTRACT FAILED: {exc!r}", flush=True)

        # Attach local consensus accuracy + delta if available.
        local_arch = per_candidate_local.get(slug, {}).get(arch)
        if local_arch is not None:
            record["consensus_oracle_accuracy_rescored"] = local_arch.get(
                "accuracy_vs_consensus"
            )
            record["consensus_oracle_n_correct"] = local_arch.get("n_correct")
            record["consensus_oracle_n_parsed"] = local_arch.get("n_parsed")
            record["consensus_oracle_n_questions"] = local_arch.get("n_questions")
            cb = record.get("scoring", {}).get("calibration_accuracy")
            local = local_arch.get("accuracy_vs_consensus")
            if cb is not None and local is not None:
                record["delta_codabench_minus_consensus"] = round(cb - local, 4)

        submissions.append(record)

        # Save incremental progress after every submission.
        _flush(started, cal_qids, submissions)

    final = _flush(started, cal_qids, submissions)
    _write_markdown(final)
    print(f"\nwrote JSON to {OUT_JSON}")
    print(f"wrote markdown to {OUT_MD}")
    return 0


def _flush(started: str, cal_qids: list[str], submissions: list[dict]) -> dict:
    """Compute summary stats and persist."""
    deltas = [
        s.get("delta_codabench_minus_consensus")
        for s in submissions
        if isinstance(s.get("delta_codabench_minus_consensus"), (int, float))
    ]
    deltas_sorted = sorted(deltas)
    n = len(deltas_sorted)
    median = (
        deltas_sorted[n // 2]
        if n and n % 2
        else ((deltas_sorted[n // 2 - 1] + deltas_sorted[n // 2]) / 2 if n else None)
    )
    mean = sum(deltas_sorted) / n if n else None
    summary = {
        "n_submissions_total": len(submissions),
        "n_submissions_with_codabench_score": sum(
            1
            for s in submissions
            if isinstance(s.get("scoring", {}).get("calibration_accuracy"), (int, float))
        ),
        "n_deltas_computed": n,
        "median_delta": round(median, 4) if median is not None else None,
        "mean_delta": round(mean, 4) if mean is not None else None,
        "deltas_distribution": {
            "abs_le_0_05": sum(1 for d in deltas if abs(d) <= 0.05),
            "abs_0_05_to_0_15": sum(1 for d in deltas if 0.05 < abs(d) <= 0.15),
            "abs_gt_0_15": sum(1 for d in deltas if abs(d) > 0.15),
        },
    }

    payload = {
        "submitted_at": started,
        "calibration_pool_qids": cal_qids,
        "submissions": submissions,
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_markdown(payload: dict) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    subs = payload["submissions"]

    # Pivot: rows = candidate_short, cols = arch, cell = "cb / cons (delta)"
    candidates: list[str] = []
    seen: set[str] = set()
    for s in subs:
        short = s.get("candidate_short")
        if short and short not in seen:
            seen.add(short)
            candidates.append(short)

    def cell(cand: str, arch: str) -> str:
        for s in subs:
            if s.get("candidate_short") == cand and s.get("architecture") == arch:
                cb = s.get("scoring", {}).get("calibration_accuracy")
                local = s.get("consensus_oracle_accuracy_rescored")
                delta = s.get("delta_codabench_minus_consensus")
                if cb is None and local is None:
                    return "n/a"
                cb_s = f"{cb:.2f}" if isinstance(cb, (int, float)) else "?"
                local_s = f"{local:.2f}" if isinstance(local, (int, float)) else "?"
                delta_s = (
                    f"{delta:+.2f}" if isinstance(delta, (int, float)) else "?"
                )
                return f"{cb_s} / {local_s} ({delta_s})"
        return "—"

    lines: list[str] = []
    lines.append("# Codabench-gold vs Consensus-oracle — Phase G re-score (2026-05-16)")
    lines.append("")
    lines.append("Notes-only working document; not committed to the paper tree.")
    lines.append("")
    lines.append("Cell format: `codabench_acc / consensus_acc (delta)` where")
    lines.append("delta = codabench - consensus. Both are calibration-pool-only")
    lines.append("(n=20 questions, 5 per novel across B01/B08/B41/B50).")
    lines.append("")
    header = "| Candidate | " + " | ".join(ARCHS) + " |"
    sep = "| --- |" + " --- |" * len(ARCHS)
    lines.append(header)
    lines.append(sep)
    for cand in candidates:
        row = "| " + cand + " | " + " | ".join(cell(cand, a) for a in ARCHS) + " |"
        lines.append(row)
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Submissions attempted: {summary['n_submissions_total']}")
    lines.append(
        f"- Submissions with a recoverable Codabench MC score: "
        f"{summary['n_submissions_with_codabench_score']}"
    )
    lines.append(f"- Deltas computable: {summary['n_deltas_computed']}")
    lines.append(f"- Median delta (codabench - consensus): {summary['median_delta']}")
    lines.append(f"- Mean delta: {summary['mean_delta']}")
    lines.append("")
    dd = summary["deltas_distribution"]
    lines.append(f"- |delta| <= 0.05: {dd['abs_le_0_05']}")
    lines.append(f"- 0.05 < |delta| <= 0.15: {dd['abs_0_05_to_0_15']}")
    lines.append(f"- |delta| > 0.15: {dd['abs_gt_0_15']}")
    lines.append("")
    lines.append("## Per-architecture median delta")
    lines.append("")
    lines.append("| Architecture | n | median delta | mean delta |")
    lines.append("| --- | --- | --- | --- |")
    for arch in ARCHS:
        deltas = [
            s.get("delta_codabench_minus_consensus")
            for s in subs
            if s.get("architecture") == arch
            and isinstance(s.get("delta_codabench_minus_consensus"), (int, float))
        ]
        if not deltas:
            lines.append(f"| {arch} | 0 | — | — |")
            continue
        ds = sorted(deltas)
        m = (
            ds[len(ds) // 2]
            if len(ds) % 2
            else (ds[len(ds) // 2 - 1] + ds[len(ds) // 2]) / 2
        )
        avg = sum(ds) / len(ds)
        lines.append(f"| {arch} | {len(ds)} | {m:+.4f} | {avg:+.4f} |")
    lines.append("")
    lines.append("## Methodology assessment")
    lines.append("")
    if summary["median_delta"] is None:
        lines.append("No deltas computable — the assessment is deferred.")
    else:
        absmed = abs(summary["median_delta"])
        if absmed <= 0.05:
            interp = (
                "Consensus oracle and Codabench-gold agree within 5 percentage "
                "points at the median, so the oracle is a defensible proxy for "
                "the gold labels at the calibration-pool scale (n=20)."
            )
        elif absmed <= 0.15:
            interp = (
                "Consensus oracle drifts from Codabench-gold by 5-15 percentage "
                "points at the median. It remains useful as an internal "
                "discriminator across candidates/architectures but should not "
                "be reported as a substitute for the leaderboard number."
            )
        else:
            interp = (
                "Consensus oracle deviates from Codabench-gold by more than 15 "
                "percentage points at the median, which means the oracle and "
                "the gold disagree often enough that its use as an accuracy "
                "proxy must be re-examined. The direction of the bias (sign of "
                "the delta) tells us whether the consensus is harsher or more "
                "lenient than the gold."
            )
        lines.append(interp)
    lines.append("")
    lines.append("## Operational notes")
    lines.append("")
    lines.append(
        "- Codabench's MC-only scoring container crashes on `cr_gen_score`. "
        "Submissions land in `Failed` status, but the per-novel T/F strings "
        "for the MC subtask are recoverable from `scoring_stdout`. The accuracy "
        "numbers in this table are extracted from that log."
    )
    lines.append(
        "- Calibration accuracy is sliced to the 20 calibration-pool QID "
        "positions per novel; the other 2,285 question slots in the submission "
        "are filled with the placeholder letter 'A' as documented in "
        "`pilot.codabench.format`."
    )
    lines.append(
        "- Submissions were throttled to 1 per 60 seconds to be friendly to "
        "the shared Codabench scoring queue."
    )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
