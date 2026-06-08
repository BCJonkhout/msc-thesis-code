"""Tests for pilot.codabench.idempotency."""
from __future__ import annotations

import json
from pathlib import Path

from pilot.codabench import idempotency as idem


def test_job_key():
    assert idem.job_key({"candidate": "c", "architecture": "raptor"}) == ("c", "raptor")
    assert idem.job_key({}) == ("", "")


def test_load_prior_absent_file_is_empty(tmp_path: Path):
    assert idem.load_prior_records(tmp_path / "nope.json") == {}


def test_load_prior_corrupt_file_is_empty(tmp_path: Path):
    p = tmp_path / "out.json"
    p.write_text("{not json", encoding="utf-8")
    assert idem.load_prior_records(p) == {}


def test_load_prior_merges_submissions_and_failures(tmp_path: Path):
    p = tmp_path / "out.json"
    p.write_text(json.dumps({
        "submissions": [{"candidate": "c1", "architecture": "raptor",
                         "submission_id": 11}],
        "failures": [{"candidate": "c2", "architecture": "graphrag",
                      "error": "boom"}],
    }), encoding="utf-8")
    prior = idem.load_prior_records(p)
    assert prior[("c1", "raptor")]["submission_id"] == 11
    assert prior[("c2", "graphrag")]["error"] == "boom"


def test_already_submitted():
    assert idem.already_submitted({"submission_id": 5}) is True
    assert idem.already_submitted({"submission_id": None}) is False
    assert idem.already_submitted({"error": "x"}) is False
    assert idem.already_submitted(None) is False


def test_split_results():
    results = {
        ("c1", "raptor"): {"candidate": "c1", "architecture": "raptor",
                           "submission_id": 1},
        ("c2", "raptor"): {"candidate": "c2", "architecture": "raptor",
                           "submission_id": None, "error": "fail"},
        ("c3", "raptor"): {"candidate": "c3", "architecture": "raptor",
                           "submission_id": 2, "error": "late error"},
    }
    subs, fails = idem.split_results(results)
    sub_keys = {(r["candidate"]) for r in subs}
    fail_keys = {(r["candidate"]) for r in fails}
    assert sub_keys == {"c1"}            # id + no error
    assert fail_keys == {"c2", "c3"}     # no id, or id with error


def test_atomic_write_json_roundtrip_and_overwrite(tmp_path: Path):
    p = tmp_path / "out.json"
    idem.atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
    # Overwrite leaves no stray tmp file.
    idem.atomic_write_json(p, {"a": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2}
    assert not (tmp_path / "out.json.tmp").exists()


def test_rerun_skips_already_submitted_keeps_recovered_scores(tmp_path: Path):
    """Integration of the helpers: a re-run skips done jobs and preserves
    their recovered scores while retrying the failed one."""
    out = tmp_path / "out.json"
    # Prior run: c1 succeeded (with a recovered score), c2 failed.
    idem.atomic_write_json(out, {
        "submissions": [{"candidate": "c1", "architecture": "raptor",
                         "submission_id": 11, "codabench_accuracy": 0.8}],
        "failures": [{"candidate": "c2", "architecture": "raptor",
                      "error": "429"}],
    })
    jobs = [{"candidate": "c1", "architecture": "raptor"},
            {"candidate": "c2", "architecture": "raptor"}]

    results = idem.load_prior_records(out)
    pending = [j for j in jobs
               if not idem.already_submitted(results.get(idem.job_key(j)))]
    # Only the failed c2 is retried; c1 is skipped.
    assert pending == [{"candidate": "c2", "architecture": "raptor"}]
    # c1's recovered score survives in the merged results.
    assert results[("c1", "raptor")]["codabench_accuracy"] == 0.8
