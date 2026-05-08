"""Tests for the --resume-from logic in pilot.cli.step_3_dry_run.

Coverage:

  - _load_resume_state correctly reads a prior run dir's
    <arch>_predictions.jsonl files into the in-memory state.
  - The completed-keys set carries (architecture, paper_id,
    question_id) tuples for every prior row.
  - Per-arch scores are seeded from the prior rows so the final
    macro aggregate reflects all questions, not only newly-run ones.
  - Missing per-arch JSONL files in the resume_from dir are
    handled gracefully (no crash, just empty seed for that arch).
  - The dispatcher loop SKIPS items whose key is already in the
    completed set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.architectures import ArchitectureResult
from pilot.cli import step_3_dry_run as cli


# ──────────────────────────────────────────────────────────────────────
# _load_resume_state
# ──────────────────────────────────────────────────────────────────────

def _write_prior_predictions(
    run_dir: Path,
    arch: str,
    rows: list[dict[str, Any]],
) -> None:
    """Helper: write a prior-run <arch>_predictions.jsonl on disk."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / f"{arch}_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


class TestLoadResumeState:
    def test_no_resume_path_returns_empty_state(self):
        per_arch, scores, completed = cli._load_resume_state(None, ["raptor"])
        assert per_arch == {}
        assert scores == {}
        assert completed == set()

    def test_loads_prior_rows_for_one_arch(self, tmp_path: Path):
        run_dir = tmp_path / "run-prior"
        _write_prior_predictions(
            run_dir,
            "raptor",
            [
                {"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
                 "predicted_answer": "A1", "answer_f1": 1.0, "evidence_f1": 0.0},
                {"dataset": "qasper", "paper_id": "p2", "question_id": "q2",
                 "predicted_answer": "A2", "answer_f1": 0.5, "evidence_f1": 0.0},
            ],
        )
        per_arch, scores, completed = cli._load_resume_state(run_dir, ["raptor"])
        assert len(per_arch["raptor"]) == 2
        assert ("raptor", "p1", "q1") in completed
        assert ("raptor", "p2", "q2") in completed
        # Scores are seeded with the float-valued metrics only.
        assert scores["raptor"]["answer_f1"] == [1.0, 0.5]
        assert scores["raptor"]["evidence_f1"] == [0.0, 0.0]

    def test_loads_multiple_archs(self, tmp_path: Path):
        run_dir = tmp_path / "run-multi"
        _write_prior_predictions(
            run_dir, "flat",
            [{"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
              "answer_f1": 0.5}],
        )
        _write_prior_predictions(
            run_dir, "naive_rag",
            [{"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
              "answer_f1": 0.6}],
        )
        per_arch, scores, completed = cli._load_resume_state(
            run_dir, ["flat", "naive_rag"]
        )
        assert len(per_arch["flat"]) == 1
        assert len(per_arch["naive_rag"]) == 1
        assert ("flat", "p1", "q1") in completed
        assert ("naive_rag", "p1", "q1") in completed

    def test_missing_arch_file_handled_gracefully(self, tmp_path: Path):
        """Resume from a run dir that only has flat predictions, but
        we ask to resume both flat and naive_rag. The missing
        naive_rag file should NOT crash — just leave naive_rag's
        seed empty so every naive_rag item runs fresh."""
        run_dir = tmp_path / "run-missing"
        _write_prior_predictions(
            run_dir, "flat",
            [{"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
              "answer_f1": 0.5}],
        )
        per_arch, scores, completed = cli._load_resume_state(
            run_dir, ["flat", "naive_rag"]
        )
        assert ("flat", "p1", "q1") in completed
        # naive_rag has no prior rows.
        assert "naive_rag" not in per_arch or per_arch["naive_rag"] == []
        assert not any(c[0] == "naive_rag" for c in completed)

    def test_only_numeric_metrics_seeded_into_scores(self, tmp_path: Path):
        """Rows can carry None / dict / non-numeric values for some
        metrics (e.g., NovelQA rows have answer_f1=None,
        predicted_letter='A'). The scores aggregator should only
        capture int/float values."""
        run_dir = tmp_path / "run-mixed"
        _write_prior_predictions(
            run_dir, "flat",
            [
                {"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
                 "answer_f1": 0.5, "evidence_f1": 0.0, "accuracy": None},
                {"dataset": "novelqa", "paper_id": "B01", "question_id": "Q1",
                 "answer_f1": None, "evidence_f1": None, "accuracy": None,
                 "predicted_letter": "B"},
            ],
        )
        _, scores, completed = cli._load_resume_state(run_dir, ["flat"])
        # Only the QASPER row's two floats should be captured.
        assert scores["flat"]["answer_f1"] == [0.5]
        assert scores["flat"]["evidence_f1"] == [0.0]
        assert "predicted_letter" not in scores["flat"]
        # Both rows still appear in completed (for SKIP semantics).
        assert ("flat", "p1", "q1") in completed
        assert ("flat", "B01", "Q1") in completed


# ──────────────────────────────────────────────────────────────────────
# Dispatcher SKIP behaviour
# ──────────────────────────────────────────────────────────────────────

class TestResumeSkipsCompletedItems:
    """Verifies the SKIP path inside run_dry_run honours the
    completed-keys set built from --resume-from. We end-to-end
    invoke run_dry_run with mocked runners so this test never
    touches Ollama or any LLM."""

    def test_completed_items_are_skipped(self, tmp_path: Path, monkeypatch):
        # 1. Build a tiny fake QASPER pool with 3 questions on disk.
        data_root = tmp_path / "data"
        qasper = data_root / "qasper"
        qasper.mkdir(parents=True)
        # Three papers, one Q each, each with a non-empty gold answer.
        papers = []
        for i in range(3):
            pid = f"p{i}"
            qid = f"q{i}"
            papers.append({
                "paper_id": pid, "title": f"paper {pid}",
                "abstract": "x",
                "full_text": [{"section_name": "s", "paragraphs": ["text"]}],
                "qas": [{
                    "question_id": qid, "question": "?",
                    "answers": [{"answer": {"free_form_answer": "yes",
                                            "evidence": [], "highlighted_evidence": [],
                                            "extractive_spans": [], "yes_no": True,
                                            "unanswerable": False}}],
                }],
            })
        (qasper / "dev.jsonl").write_text(
            "\n".join(json.dumps(p) for p in papers) + "\n", encoding="utf-8"
        )
        (qasper / "calibration_pool.jsonl").write_text(
            "\n".join(json.dumps({"paper_id": p["paper_id"],
                                  "question_id": p["qas"][0]["question_id"],
                                  "question": "?"}) for p in papers) + "\n",
            encoding="utf-8",
        )

        # 2. Build a prior run dir with 2 of 3 questions already done.
        prior = tmp_path / "outputs" / "runs" / "prior"
        _write_prior_predictions(
            prior, "flat",
            [
                {"dataset": "qasper", "paper_id": "p0", "question_id": "q0",
                 "answer_f1": 0.5, "evidence_f1": 0.0},
                {"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
                 "answer_f1": 0.7, "evidence_f1": 0.0},
            ],
        )

        # 3. Mock everything that would touch the network.
        invocations: list[tuple[str, str]] = []

        def fake_run_flat(**kwargs):
            invocations.append((kwargs["query"], kwargs["document"][:5]))
            return ArchitectureResult(architecture="flat", predicted_answer="z")

        monkeypatch.setattr(cli, "run_flat", fake_run_flat)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())
        # Ensure no embedder is actually constructed (flat doesn't need one,
        # but the orchestrator might still try if any non-flat arch is in
        # the list). We're only running flat here.

        # 4. Drive run_dry_run with resume_from set.
        out = cli.run_dry_run(
            architectures=["flat"],
            datasets=["qasper"],
            answerer_provider="gemini",
            answerer_model="m",
            embedder_model="bge-m3",
            naive_rag_top_k=8,
            data_root=data_root,
            out_dir=tmp_path / "out",
            resume_from=prior,
        )

        # 5. Verify only the missing item (p2, q2) was actually invoked.
        assert len(invocations) == 1
        # Verify the verdict counts all 3 (2 prior + 1 new).
        assert out["per_arch_counts"]["flat"] == 3
