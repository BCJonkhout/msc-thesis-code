"""Idempotency helpers for Codabench submission drivers.

Re-running a submission driver after a crash (or to fill in a few failed
cells) must NOT create duplicate Codabench submissions -- the submission
is throttled / quota-limited and the MC scoring container is flaky -- and
must NOT discard already-recovered gold scores by overwriting the output
JSON from an empty start. These helpers let a driver load prior results,
skip jobs that already carry a submission id, merge new outcomes in, and
write the accumulated result atomically.

Records are keyed by (candidate, architecture); a driver builds one
record per job and merges it into a results dict, then splits the dict
into submissions / failures for reporting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def job_key(rec: dict[str, Any]) -> tuple[str, str]:
    """Stable key for a submission record / job: (candidate, architecture)."""
    return (rec.get("candidate", ""), rec.get("architecture", ""))


def load_prior_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load prior submission + failure records keyed by (candidate, arch).

    Returns {} when the file is absent or unreadable, so a first run or a
    corrupt prior file simply starts fresh rather than crashing.
    """
    if not path.exists():
        return {}
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in (prior.get("submissions") or []) + (prior.get("failures") or []):
        out[job_key(rec)] = rec
    return out


def already_submitted(rec: dict[str, Any] | None) -> bool:
    """True when a prior record already holds a Codabench submission id.

    The submission is the throttled / quota-consuming step. If it
    succeeded, never re-submit -- re-extracting a score from the existing
    submission_id does not require a new submission. A failed job (no
    submission_id) is NOT skipped, so it is retried on the next run.
    """
    return bool(rec and rec.get("submission_id"))


def split_results(
    results: dict[tuple[str, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition a results dict into (submissions, failures).

    A record counts as a submission when it has a submission_id and no
    top-level error; everything else is a failure (to be retried).
    """
    submissions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rec in results.values():
        if rec.get("submission_id") and not rec.get("error"):
            submissions.append(rec)
        else:
            failures.append(rec)
    return submissions, failures


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via tmp + fsync + os.replace so a crash mid-write cannot
    corrupt or truncate the accumulated results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
    except (OSError, AttributeError):
        pass
    os.replace(tmp, path)
