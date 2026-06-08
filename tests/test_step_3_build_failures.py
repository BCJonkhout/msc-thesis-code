"""Tests for build-failure handling in pilot.cli.step_3_dry_run.

A RAPTOR/GraphRAG build can fail (clustering, an embedder outage, a
pathological document). Two failure modes must both be handled so the
failure is VISIBLE and COUNTED rather than silently thinning the
architecture's denominator:

  - soft failure: the runner returns ArchitectureResult(failed=True)
    without raising (RAPTOR does this today);
  - hard failure: the runner raises.

In both cases the cell must be (a) scored as wrong so it stays in the
denominator (parity with Flat/Naive, which never fail to build),
(b) flagged build_failed=True in the prediction row, (c) recorded in
build_failures.jsonl, and (d) counted in the verdict's
build_failures_per_arch. A build_failed cell counts as completed, so
resume does not auto-retry a (now-deterministic) build failure forever;
the operator re-drives specific cells by removing their rows.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from pilot.architectures import ArchitectureResult
from pilot.cli import step_3_dry_run as cli


def _make_qasper_pool(data_root: Path, pairs: list[tuple[str, str]]) -> None:
    qasper = data_root / "qasper"
    qasper.mkdir(parents=True, exist_ok=True)
    papers, cal = [], []
    for pid, qid in pairs:
        papers.append({
            "paper_id": pid, "title": f"paper {pid}", "abstract": "x",
            "full_text": [{"section_name": "s", "paragraphs": ["text"]}],
            "qas": [{
                "question_id": qid, "question": "?",
                "answers": [{"answer": {"free_form_answer": "yes",
                                        "evidence": [], "highlighted_evidence": [],
                                        "extractive_spans": [], "yes_no": True,
                                        "unanswerable": False}}],
            }],
        })
        cal.append({"paper_id": pid, "question_id": qid, "question": "?"})
    (qasper / "dev.jsonl").write_text(
        "\n".join(json.dumps(p) for p in papers) + "\n", encoding="utf-8")
    (qasper / "calibration_pool.jsonl").write_text(
        "\n".join(json.dumps(r) for r in cal) + "\n", encoding="utf-8")


def _run(tmp_path: Path, run_id: str):
    return cli.run_dry_run(
        architectures=["flat"],
        datasets=["qasper"],
        answerer_provider="gemini",
        answerer_model="m",
        embedder_model="bge-m3",
        naive_rag_top_k=8,
        data_root=tmp_path / "data",
        out_dir=tmp_path / "out",
        run_id=run_id,
        runs_root=tmp_path / "runs",
    )


def test_soft_failure_is_counted_flagged_and_logged(tmp_path: Path, monkeypatch):
    _make_qasper_pool(tmp_path / "data", [("p0", "q0")])

    def fake_run_flat(**kwargs):
        return ArchitectureResult(
            architecture="flat", predicted_answer="",
            failed=True, failure_reason="cluster build failed",
        )

    monkeypatch.setattr(cli, "run_flat", fake_run_flat)
    monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

    out = _run(tmp_path, "softfail")

    # Counted in the denominator, not dropped.
    assert out["per_arch_counts"]["flat"] == 1
    assert out["build_failures_total"] == 1
    assert out["build_failures_per_arch"]["flat"] == 1

    run_dir = tmp_path / "runs" / "softfail"
    # build_failures.jsonl records the reason.
    bf = [json.loads(ln) for ln in
          (run_dir / "build_failures.jsonl").read_text(encoding="utf-8").splitlines()
          if ln.strip()]
    assert len(bf) == 1
    assert bf[0]["failure_reason"] == "cluster build failed"
    assert bf[0]["paper_id"] == "p0"
    # Prediction row is flagged.
    pred = json.loads(
        (run_dir / "flat_predictions.jsonl").read_text(encoding="utf-8").strip())
    assert pred["build_failed"] is True


def test_raised_failure_is_counted_not_dropped(tmp_path: Path, monkeypatch):
    _make_qasper_pool(tmp_path / "data", [("p0", "q0")])

    def fake_run_flat(**kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "run_flat", fake_run_flat)
    monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

    out = _run(tmp_path, "raisefail")

    # The raise is recorded in failures AND counted as a build failure,
    # and a row is still written (not silently dropped).
    assert out["failures_count"] == 1
    assert out["build_failures_total"] == 1
    assert out["per_arch_counts"]["flat"] == 1
    pred = json.loads(
        (tmp_path / "runs" / "raisefail" / "flat_predictions.jsonl")
        .read_text(encoding="utf-8").strip())
    assert pred["build_failed"] is True


def test_build_failed_cell_counts_as_completed_on_resume(tmp_path: Path, monkeypatch):
    """A build_failed row is treated as completed: resume does not
    auto-retry a deterministic build failure."""
    _make_qasper_pool(tmp_path / "data", [("p0", "q0")])
    run_dir = tmp_path / "runs" / "resumefail"
    run_dir.mkdir(parents=True)
    (run_dir / "flat_predictions.jsonl").write_text(
        json.dumps({"dataset": "qasper", "paper_id": "p0", "question_id": "q0",
                    "run_index": 0, "predicted_answer": "",
                    "build_failed": True, "answer_f1": 0.0}) + "\n",
        encoding="utf-8")

    calls: list[int] = []

    def fake_run_flat(**kwargs):
        calls.append(1)
        return ArchitectureResult(architecture="flat", predicted_answer="z")

    monkeypatch.setattr(cli, "run_flat", fake_run_flat)
    monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

    _run(tmp_path, "resumefail")
    assert calls == []  # the build_failed cell was skipped on resume
