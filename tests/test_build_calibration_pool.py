"""Tests for pilot.data.build_calibration_pool.

Coverage:

  - _has_evidence correctly detects QASPER's evidence /
    highlighted_evidence annotations at the answer level.
  - build_qasper_pool samples deterministically (same seed → same
    pool), excludes papers that also appear in the test split,
    rejects questions without any evidence-bearing answer, and
    picks one question per paper.
  - build_novelqa_pool samples 4 calibration novels × 5 questions
    deterministically, excludes B48 (the outlier stress test),
    and writes the calibration_novels.json exclusion list.

The tests use synthetic JSONL fixtures on a tmp_path data root so
they do not require the real datasets to be downloaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot.data.build_calibration_pool import (
    _QASPER_POOL_SIZE,
    _has_evidence,
    build_novelqa_pool,
    build_qasper_pool,
)


# ──────────────────────────────────────────────────────────────────────
# _has_evidence
# ──────────────────────────────────────────────────────────────────────

class TestHasEvidence:
    def test_evidence_only(self):
        ann = {"answer": {"evidence": ["sentence one"]}}
        assert _has_evidence(ann) is True

    def test_highlighted_evidence_only(self):
        ann = {"answer": {"highlighted_evidence": ["sentence one"]}}
        assert _has_evidence(ann) is True

    def test_both_empty(self):
        ann = {"answer": {"evidence": [], "highlighted_evidence": []}}
        assert _has_evidence(ann) is False

    def test_no_answer_field(self):
        assert _has_evidence({}) is False

    def test_answer_not_a_dict(self):
        assert _has_evidence({"answer": "not a dict"}) is False


# ──────────────────────────────────────────────────────────────────────
# QASPER pool fixture
# ──────────────────────────────────────────────────────────────────────

def _make_qasper_paper(paper_id: str, *, n_questions: int, with_evidence: bool):
    """Build a synthetic QASPER paper record with `n_questions` QAs."""
    qas = []
    for i in range(n_questions):
        qid = f"q_{paper_id}_{i:02d}"
        if with_evidence:
            answers = [
                {"answer": {"evidence": ["snippet"], "highlighted_evidence": ["s1"]}},
            ]
        else:
            answers = [{"answer": {"evidence": [], "highlighted_evidence": []}}]
        qas.append({"question_id": qid, "question": f"q text {i}", "answers": answers})
    return {"paper_id": paper_id, "qas": qas, "title": f"paper {paper_id}"}


@pytest.fixture
def qasper_root(tmp_path: Path) -> Path:
    """Build a synthetic QASPER layout: 30 evidence-bearing dev
    papers, 5 non-evidence dev papers, 5 dev papers that overlap
    with the test split (must be excluded)."""
    qasper = tmp_path / "qasper"
    qasper.mkdir(parents=True)

    dev = []
    # 30 evidence-bearing, distinct from test.
    for i in range(30):
        dev.append(_make_qasper_paper(f"e{i:03d}", n_questions=2, with_evidence=True))
    # 5 with no evidence at all.
    for i in range(5):
        dev.append(_make_qasper_paper(f"n{i:03d}", n_questions=2, with_evidence=False))
    # 5 that ALSO appear in test (must be excluded from the pool).
    for i in range(5):
        dev.append(_make_qasper_paper(f"t{i:03d}", n_questions=2, with_evidence=True))

    # Test split shadows the t* papers.
    test = [_make_qasper_paper(f"t{i:03d}", n_questions=1, with_evidence=False) for i in range(5)]

    with (qasper / "dev.jsonl").open("w", encoding="utf-8") as fh:
        for p in dev:
            fh.write(json.dumps(p))
            fh.write("\n")
    with (qasper / "test.jsonl").open("w", encoding="utf-8") as fh:
        for p in test:
            fh.write(json.dumps(p))
            fh.write("\n")
    return tmp_path


class TestBuildQasperPool:
    def test_pool_has_exactly_the_target_size(self, qasper_root: Path):
        out = build_qasper_pool(qasper_root, seed=42)
        assert out["status"] == "built"
        assert out["pool_size"] == _QASPER_POOL_SIZE
        # candidate_papers counts all papers that pass the
        # evidence-bearing + not-in-test filter (30 here).
        assert out["candidate_papers"] == 30

    def test_excludes_test_overlap_and_no_evidence_papers(self, qasper_root: Path):
        build_qasper_pool(qasper_root, seed=42)
        pool = [
            json.loads(l)
            for l in (qasper_root / "qasper" / "calibration_pool.jsonl")
            .open(encoding="utf-8")
            if l.strip()
        ]
        # No paper id starting with "t" (test-shadowed) or "n" (no evidence).
        assert all(not p["paper_id"].startswith(("t", "n")) for p in pool)

    def test_one_question_per_paper(self, qasper_root: Path):
        build_qasper_pool(qasper_root, seed=42)
        pool = [
            json.loads(l)
            for l in (qasper_root / "qasper" / "calibration_pool.jsonl")
            .open(encoding="utf-8")
            if l.strip()
        ]
        paper_ids = [p["paper_id"] for p in pool]
        assert len(paper_ids) == len(set(paper_ids))

    def test_deterministic_under_same_seed(self, qasper_root: Path, tmp_path: Path):
        # Build the same pool twice and verify identical output.
        build_qasper_pool(qasper_root, seed=42)
        first = (qasper_root / "qasper" / "calibration_pool.jsonl").read_text(
            encoding="utf-8"
        )
        # Move the result aside and rebuild.
        (qasper_root / "qasper" / "calibration_pool.jsonl").rename(
            tmp_path / "first_pool.jsonl"
        )
        build_qasper_pool(qasper_root, seed=42)
        second = (qasper_root / "qasper" / "calibration_pool.jsonl").read_text(
            encoding="utf-8"
        )
        assert first == second

    def test_different_seeds_produce_different_pools(self, qasper_root: Path):
        build_qasper_pool(qasper_root, seed=42)
        pool42 = (qasper_root / "qasper" / "calibration_pool.jsonl").read_text(
            encoding="utf-8"
        )
        build_qasper_pool(qasper_root, seed=7)
        pool7 = (qasper_root / "qasper" / "calibration_pool.jsonl").read_text(
            encoding="utf-8"
        )
        assert pool42 != pool7

    def test_too_few_candidates_returns_error(self, tmp_path: Path):
        qasper = tmp_path / "qasper"
        qasper.mkdir(parents=True)
        # Only 5 evidence-bearing papers; pool target is 20.
        dev = [
            _make_qasper_paper(f"e{i:03d}", n_questions=1, with_evidence=True)
            for i in range(5)
        ]
        with (qasper / "dev.jsonl").open("w", encoding="utf-8") as fh:
            for p in dev:
                fh.write(json.dumps(p))
                fh.write("\n")
        out = build_qasper_pool(tmp_path, seed=42)
        assert out["status"] == "error"
        assert "5" in out["error"] and "20" in out["error"]


# ──────────────────────────────────────────────────────────────────────
# NovelQA pool fixture
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def novelqa_root(tmp_path: Path) -> Path:
    """Build a synthetic NovelQA layout: 10 novels (B00..B09) each
    with 8 questions on disk + texts under full_texts/."""
    novelqa = tmp_path / "novelqa"
    full_texts = novelqa / "full_texts"
    full_texts.mkdir(parents=True)

    questions = []
    for i in range(10):
        bid = f"B{i:02d}"
        (full_texts / f"{bid}.txt").write_text(f"text of {bid}", encoding="utf-8")
        for j in range(8):
            questions.append({"novel_id": bid, "question_id": f"Q{bid}_{j:02d}"})
    # Add B48 (outlier stress test) — should be excluded by the
    # builder regardless of seed.
    (full_texts / "B48.txt").write_text("history of rome", encoding="utf-8")
    for j in range(8):
        questions.append({"novel_id": "B48", "question_id": f"QB48_{j:02d}"})

    with (novelqa / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q))
            fh.write("\n")
    return tmp_path


class TestBuildNovelqaPool:
    def test_pool_size_is_4x5(self, novelqa_root: Path):
        out = build_novelqa_pool(novelqa_root, seed=42)
        assert out["status"] == "built"
        assert out["pool_size"] == 20
        assert len(out["calibration_novels"]) == 4

    def test_excludes_b48(self, novelqa_root: Path):
        out = build_novelqa_pool(novelqa_root, seed=42)
        # B48 must never be picked, regardless of seed.
        assert "B48" not in out["calibration_novels"]
        cal = [
            json.loads(l)
            for l in (novelqa_root / "novelqa" / "calibration_pool.jsonl")
            .open(encoding="utf-8")
            if l.strip()
        ]
        assert all(c["novel_id"] != "B48" for c in cal)

    def test_calibration_novels_recorded_for_main_eval_exclusion(
        self, novelqa_root: Path
    ):
        build_novelqa_pool(novelqa_root, seed=42)
        record = json.loads(
            (novelqa_root / "novelqa" / "calibration_novels.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(record["calibration_novels"]) == 4
        # B48 always appears in the held-out exclusion list (so the
        # main evaluation skips it as the outlier stress test).
        assert "B48" in record["exclude_from_main_eval"]
        # The 4 calibration novels are also in the exclusion list.
        for nid in record["calibration_novels"]:
            assert nid in record["exclude_from_main_eval"]

    def test_deterministic_under_same_seed(
        self, novelqa_root: Path, tmp_path: Path
    ):
        build_novelqa_pool(novelqa_root, seed=42)
        first = (novelqa_root / "novelqa" / "calibration_pool.jsonl").read_text(
            encoding="utf-8"
        )
        (novelqa_root / "novelqa" / "calibration_pool.jsonl").rename(
            tmp_path / "first_pool.jsonl"
        )
        build_novelqa_pool(novelqa_root, seed=42)
        second = (novelqa_root / "novelqa" / "calibration_pool.jsonl").read_text(
            encoding="utf-8"
        )
        assert first == second

    def test_too_few_eligible_novels_returns_error(self, tmp_path: Path):
        novelqa = tmp_path / "novelqa"
        full_texts = novelqa / "full_texts"
        full_texts.mkdir(parents=True)
        # Only 2 novels with enough questions; pool needs 4.
        questions = []
        for i in range(2):
            bid = f"B{i:02d}"
            (full_texts / f"{bid}.txt").write_text("x", encoding="utf-8")
            for j in range(8):
                questions.append({"novel_id": bid, "question_id": f"Q{bid}_{j}"})
        with (novelqa / "questions.jsonl").open("w", encoding="utf-8") as fh:
            for q in questions:
                fh.write(json.dumps(q))
                fh.write("\n")
        out = build_novelqa_pool(tmp_path, seed=42)
        assert out["status"] == "error"
        assert "4" in out["error"]
