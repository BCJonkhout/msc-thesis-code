"""Tests for the full main-study split loaders in step_3_dry_run.

Key properties:
  - calibration questions are excluded (no calibration/eval leakage);
  - papers/novels with < min_queries are dropped (repeated-context
    cost is undefined for a single-query document);
  - B48 is excluded from NovelQA;
  - max_docs caps to the first N ELIGIBLE docs in sorted-id order, so a
    validation slice is a deterministic PREFIX of the full split (this is
    what lets the full run resume-in-place over the slice's cells).
"""
from __future__ import annotations

import json
from pathlib import Path

from pilot.cli import step_3_dry_run as cli


def _write_qasper(data_root: Path):
    qasper = data_root / "qasper"
    qasper.mkdir(parents=True)

    def qa(qid):
        return {"question_id": qid, "question": "?",
                "answers": [{"answer": {"free_form_answer": "yes",
                                        "highlighted_evidence": []}}]}

    papers = [
        {"paper_id": "pA", "title": "A", "abstract": "x",
         "full_text": [{"section_name": "s", "paragraphs": ["t"]}],
         "qas": [qa("qA1"), qa("qA2"), qa("qA3")]},
        {"paper_id": "pB", "title": "B", "abstract": "x",
         "full_text": [{"section_name": "s", "paragraphs": ["t"]}],
         "qas": [qa("qB1")]},  # single question -> ineligible
        {"paper_id": "pC", "title": "C", "abstract": "x",
         "full_text": [{"section_name": "s", "paragraphs": ["t"]}],
         "qas": [qa("qC1"), qa("qC2")]},
    ]
    (qasper / "dev.jsonl").write_text(
        "\n".join(json.dumps(p) for p in papers) + "\n", encoding="utf-8")
    # Calibration excludes qA2.
    (qasper / "calibration_pool.jsonl").write_text(
        json.dumps({"paper_id": "pA", "question_id": "qA2", "question": "?"}) + "\n",
        encoding="utf-8")


def _write_novelqa(data_root: Path):
    nq = data_root / "novelqa"
    ft = nq / "full_texts"
    ft.mkdir(parents=True)
    rows = [
        {"novel_id": "nA", "question_id": "qNA1", "Question": "?", "Options": {"A": "a"}},
        {"novel_id": "nA", "question_id": "qNA2", "Question": "?", "Options": {"A": "a"}},
        {"novel_id": "nA", "question_id": "qNA3", "Question": "?", "Options": {"A": "a"}},
        {"novel_id": "nB", "question_id": "qNB1", "Question": "?", "Options": {"A": "a"}},
        {"novel_id": "nB", "question_id": "qNB2", "Question": "?", "Options": {"A": "a"}},
        {"novel_id": "B48", "question_id": "qB1", "Question": "?", "Options": {"A": "a"}},
        {"novel_id": "B48", "question_id": "qB2", "Question": "?", "Options": {"A": "a"}},
    ]
    (nq / "questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (nq / "calibration_pool.jsonl").write_text(
        json.dumps({"novel_id": "nA", "question_id": "qNA2", "Question": "?"}) + "\n",
        encoding="utf-8")
    for nid in ("nA", "nB", "B48"):
        (ft / f"{nid}.txt").write_text(f"text of {nid}", encoding="utf-8")


def _keys(items):
    return {(it["paper_id"], it["question_id"]) for it in items}


def test_qasper_full_excludes_calibration_and_ineligible(tmp_path: Path):
    _write_qasper(tmp_path)
    items = cli.load_qasper_full(tmp_path)
    keys = _keys(items)
    assert ("pA", "qA2") not in keys           # calibration excluded
    assert not any(pid == "pB" for pid, _ in keys)  # single-question paper dropped
    assert keys == {("pA", "qA1"), ("pA", "qA3"), ("pC", "qC1"), ("pC", "qC2")}


def test_qasper_slice_is_prefix_of_full(tmp_path: Path):
    _write_qasper(tmp_path)
    full = _keys(cli.load_qasper_full(tmp_path))
    slice1 = _keys(cli.load_qasper_full(tmp_path, max_docs=1))
    # First eligible paper (sorted) is pA.
    assert slice1 == {("pA", "qA1"), ("pA", "qA3")}
    assert slice1 <= full  # prefix -> resume-in-place continues seamlessly


def test_novelqa_full_excludes_b48_and_calibration(tmp_path: Path):
    _write_novelqa(tmp_path)
    items = cli.load_novelqa_full(tmp_path)
    keys = _keys(items)
    assert not any(pid == "B48" for pid, _ in keys)   # B48 excluded
    assert ("nA", "qNA2") not in keys                  # calibration excluded
    assert keys == {("nA", "qNA1"), ("nA", "qNA3"), ("nB", "qNB1"), ("nB", "qNB2")}
    # Options + document are populated for MC scoring.
    assert all(it["options"] for it in items)
    assert all(it["document"] for it in items)


def test_novelqa_slice_is_prefix_of_full(tmp_path: Path):
    _write_novelqa(tmp_path)
    full = _keys(cli.load_novelqa_full(tmp_path))
    slice1 = _keys(cli.load_novelqa_full(tmp_path, max_docs=1))
    assert slice1 == {("nA", "qNA1"), ("nA", "qNA3")}  # first eligible novel
    assert slice1 <= full


def test_validation_slice_then_full_resumes_in_place(tmp_path: Path, monkeypatch):
    """The dress rehearsal must seamlessly continue: a validation slice
    (max_docs) and the full run share ONE run dir, so expanding to full
    skips the slice's cells and only runs the remainder."""
    from unittest.mock import MagicMock
    from pilot.architectures import ArchitectureResult

    _write_qasper(tmp_path)
    runs_root = tmp_path / "runs"

    invocations: list[str] = []

    def fake_run_flat(**kwargs):
        invocations.append(kwargs["query"])
        return ArchitectureResult(architecture="flat", predicted_answer="z")

    monkeypatch.setattr(cli, "run_flat", fake_run_flat)
    monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

    common = dict(
        architectures=["flat"], datasets=["qasper"],
        answerer_provider="gemini", answerer_model="m",
        embedder_model="bge-m3", naive_rag_top_k=8,
        data_root=tmp_path, out_dir=tmp_path / "out",
        runs_root=runs_root, split="full",
    )

    # Phase 1: validation slice — first 1 eligible paper (pA -> 2 questions).
    s1 = cli.run_dry_run(**common, max_docs_per_dataset={"qasper": 1})
    assert len(invocations) == 2
    slice_dir = s1["predictions_dir"]

    # Phase 2: full run (no caps) — SAME dir, resumes in place, runs only pC.
    invocations.clear()
    s2 = cli.run_dry_run(**common, max_docs_per_dataset={})
    assert s2["predictions_dir"] == slice_dir          # same run dir
    assert len(invocations) == 2                        # only pC's 2 questions ran
    assert s2["per_arch_counts"]["flat"] == 4           # pA (slice) + pC (full)
