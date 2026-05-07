"""Extract MC accuracy from a Codabench submission's scoring_stdout log.

The platform's scoring container computes per-novel per-question
correctness (T/F strings) and emits them to ``scoring_stdout``
before printing the final aggregate scores. Even when the
submission is marked Failed (e.g., the gen-subtask GPT-4 call
401s on a stub key), the MC scoring completes successfully and
the T/F strings are recoverable from the log.

This module:

  - downloads the platform's scoring_stdout for a given submission,
  - parses the per-novel T/F strings,
  - optionally slices to the calibration positions if a
    calibration pool is provided,
  - returns a dict with overall + calibration-only accuracy.

Usage::

    from pilot.codabench.extract_score import extract_mc_accuracy
    out = extract_mc_accuracy(submission_id=715402)
    print(out["overall_accuracy"])
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from pilot.codabench.submit import _build_authenticated_client


_NOVEL_LINE_RE = re.compile(r'"([a-z][^"]*)":\s*"([TF]+)"')


def _project_data_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3] / "data"


def _norm_title(title: str) -> str:
    """Codabench's per-novel keys in scoring_stdout are
    lowercase-no-punctuation versions of the human title."""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _question_order_per_novel(data_root: Path) -> dict[str, list[str]]:
    questions_path = data_root / "novelqa" / "questions.jsonl"
    out: dict[str, list[str]] = {}
    with questions_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            out.setdefault(q["novel_id"], []).append(q["question_id"])
    return out


def _calibration_indices(data_root: Path) -> dict[str, list[int]]:
    """Per-novel 0-based positions in QID order for the
    calibration questions, so we can slice the platform's
    correctness strings to just the positions where we have real
    predictions (vs placeholder fill)."""
    pool_path = data_root / "novelqa" / "calibration_pool.jsonl"
    if not pool_path.exists():
        return {}
    by_novel: dict[str, list[str]] = {}
    with pool_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            by_novel.setdefault(q["novel_id"], []).append(q["question_id"])
    orders = _question_order_per_novel(data_root)
    out: dict[str, list[int]] = {}
    for novel_id, qids in by_novel.items():
        full = orders.get(novel_id, [])
        out[novel_id] = sorted(full.index(qid) for qid in qids if qid in full)
    return out


def fetch_correctness_strings(submission_id: int) -> dict[str, str]:
    """Download a submission's scoring_stdout and parse per-novel T/F strings."""
    client = _build_authenticated_client()
    try:
        r = client.get(f"/api/submissions/{submission_id}/get_details/")
        r.raise_for_status()
        body = r.json()
        for log in body.get("logs", []):
            if log.get("name") != "scoring_stdout":
                continue
            url = log.get("data_file")
            if not url:
                continue
            log_resp = httpx.get(url, timeout=60)
            if log_resp.status_code != 200 or not log_resp.text:
                continue
            return dict(_NOVEL_LINE_RE.findall(log_resp.text))
    finally:
        client.close()
    return {}


def extract_mc_accuracy(
    submission_id: int,
    *,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Pull a Codabench submission's MC accuracy from its log.

    Returns a dict with overall accuracy + per-novel calibration
    breakdown if a calibration pool is on disk.
    """
    data_root = data_root or _project_data_root()
    strings = fetch_correctness_strings(submission_id)
    if not strings:
        return {
            "submission_id": submission_id,
            "error": "no scoring_stdout / no T-F strings recovered",
        }

    total_t = sum(s.count("T") for s in strings.values())
    total_q = sum(len(s) for s in strings.values())
    overall_accuracy = total_t / total_q if total_q else 0.0

    out: dict[str, Any] = {
        "submission_id": submission_id,
        "novels_with_results": len(strings),
        "total_questions": total_q,
        "total_correct": total_t,
        "overall_accuracy": round(overall_accuracy, 4),
    }

    cal_indices = _calibration_indices(data_root)
    bookmeta_path = data_root / "novelqa" / "bookmeta.json"
    if cal_indices and bookmeta_path.exists():
        meta = json.loads(bookmeta_path.read_text(encoding="utf-8"))
        cal_breakdown: dict[str, Any] = {}
        cal_total_t = cal_total_q = 0
        for novel_id, idxs in cal_indices.items():
            title = meta.get(novel_id, {}).get("title")
            if not title:
                continue
            s = strings.get(_norm_title(title), "")
            if not s:
                continue
            t = sum(1 for i in idxs if i < len(s) and s[i] == "T")
            n = sum(1 for i in idxs if i < len(s))
            cal_total_t += t
            cal_total_q += n
            cal_breakdown[novel_id] = {
                "title": title,
                "correct": t,
                "total": n,
                "accuracy": round(t / n, 4) if n else None,
            }
        out["calibration_breakdown"] = cal_breakdown
        out["calibration_accuracy"] = (
            round(cal_total_t / cal_total_q, 4) if cal_total_q else 0.0
        )
        out["calibration_total_correct"] = cal_total_t
        out["calibration_total_questions"] = cal_total_q

    return out


def main() -> int:
    import argparse
    import sys

    from pilot.env import load_env

    load_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--submission-id", type=int, required=True)
    args = parser.parse_args()
    result = extract_mc_accuracy(args.submission_id)
    print(json.dumps(result, indent=2))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
