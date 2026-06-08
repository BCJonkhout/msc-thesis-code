"""Tests for the resume-in-place logic in pilot.cli.step_3_dry_run.

Resume model (chosen 2026-06-08): a sweep writes into a stable run dir
derived from its configuration (or an explicit --run-id). Re-invoking
the same sweep reopens the SAME dir, skips completed
(arch, paper_id, question_id, run_index) cells, and appends only the
rest. The append-only ledger keeps every prior cost row, so a resumed
run never under-counts cost.

Coverage:

  - _load_resume_state reads a run dir's <arch>_predictions.jsonl files
    into in-memory state, keyed by the 4-tuple including run_index.
  - Per-arch scores are seeded from prior rows so the macro aggregate
    reflects all questions, not only newly-run ones.
  - Missing per-arch JSONL files are handled gracefully.
  - A torn trailing line (partial write at a crash) is tolerated.
  - The dispatcher loop SKIPS cells already on disk and resumes in
    place (same dir, append) rather than minting a new dir.
  - run_index distinguishes repeats: completing run_index 0 does not
    mark run_index 1 done.
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
    """Helper: write a run dir's <arch>_predictions.jsonl on disk."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / f"{arch}_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


class TestLoadResumeState:
    def test_no_run_dir_returns_empty_state(self):
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
                 "run_index": 0, "predicted_answer": "A1",
                 "answer_f1": 1.0, "evidence_f1": 0.0},
                {"dataset": "qasper", "paper_id": "p2", "question_id": "q2",
                 "run_index": 0, "predicted_answer": "A2",
                 "answer_f1": 0.5, "evidence_f1": 0.0},
            ],
        )
        per_arch, scores, completed = cli._load_resume_state(run_dir, ["raptor"])
        assert len(per_arch["raptor"]) == 2
        assert ("raptor", "p1", "q1", 0) in completed
        assert ("raptor", "p2", "q2", 0) in completed
        # Scores are seeded with the float-valued metrics only.
        assert scores["raptor"]["answer_f1"] == [1.0, 0.5]
        assert scores["raptor"]["evidence_f1"] == [0.0, 0.0]

    def test_missing_run_index_defaults_to_zero(self, tmp_path: Path):
        """Legacy rows without a run_index field are treated as run 0."""
        run_dir = tmp_path / "run-legacy"
        _write_prior_predictions(
            run_dir, "flat",
            [{"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
              "answer_f1": 0.5}],
        )
        _, _, completed = cli._load_resume_state(run_dir, ["flat"])
        assert ("flat", "p1", "q1", 0) in completed

    def test_run_index_distinguishes_repeats(self, tmp_path: Path):
        """Rows for the same (arch, paper, qid) at different run_index
        produce distinct completed keys."""
        run_dir = tmp_path / "run-repeats"
        _write_prior_predictions(
            run_dir, "flat",
            [
                {"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
                 "run_index": 0, "answer_f1": 0.5},
                {"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
                 "run_index": 1, "answer_f1": 0.6},
            ],
        )
        _, scores, completed = cli._load_resume_state(run_dir, ["flat"])
        assert ("flat", "p1", "q1", 0) in completed
        assert ("flat", "p1", "q1", 1) in completed
        assert ("flat", "p1", "q1", 2) not in completed
        assert scores["flat"]["answer_f1"] == [0.5, 0.6]

    def test_missing_arch_file_handled_gracefully(self, tmp_path: Path):
        run_dir = tmp_path / "run-missing"
        _write_prior_predictions(
            run_dir, "flat",
            [{"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
              "run_index": 0, "answer_f1": 0.5}],
        )
        per_arch, scores, completed = cli._load_resume_state(
            run_dir, ["flat", "naive_rag"]
        )
        assert ("flat", "p1", "q1", 0) in completed
        assert "naive_rag" not in per_arch or per_arch["naive_rag"] == []
        assert not any(c[0] == "naive_rag" for c in completed)

    def test_torn_trailing_line_is_tolerated(self, tmp_path: Path):
        """A crash can leave a half-written final JSONL line. The loader
        must skip only that last line (re-run on the next pass), not
        crash the whole resume."""
        run_dir = tmp_path / "run-torn"
        run_dir.mkdir(parents=True)
        path = run_dir / "flat_predictions.jsonl"
        good = json.dumps({"dataset": "qasper", "paper_id": "p1",
                           "question_id": "q1", "run_index": 0,
                           "answer_f1": 0.5})
        with path.open("w", encoding="utf-8") as fh:
            fh.write(good + "\n")
            fh.write('{"dataset": "qasper", "paper_id": "p2", "questio')  # torn
        per_arch, _, completed = cli._load_resume_state(run_dir, ["flat"])
        assert ("flat", "p1", "q1", 0) in completed
        assert len(per_arch["flat"]) == 1  # torn line skipped

    def test_interior_corruption_raises(self, tmp_path: Path):
        """Corruption that is NOT the last line is a real problem."""
        run_dir = tmp_path / "run-corrupt"
        run_dir.mkdir(parents=True)
        path = run_dir / "flat_predictions.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("{not valid json}\n")
            fh.write(json.dumps({"dataset": "qasper", "paper_id": "p2",
                                 "question_id": "q2", "run_index": 0}) + "\n")
        with pytest.raises(json.JSONDecodeError):
            cli._load_resume_state(run_dir, ["flat"])


# ──────────────────────────────────────────────────────────────────────
# Resume-in-place dispatcher behaviour
# ──────────────────────────────────────────────────────────────────────

class TestResumeInPlaceSkipsCompletedItems:
    """End-to-end: run_dry_run resumes IN PLACE into its run dir, skips
    completed cells, and appends only the missing ones. Mocked runners
    keep this off Ollama / any LLM."""

    def _make_pool(self, tmp_path: Path, n: int = 3) -> Path:
        data_root = tmp_path / "data"
        qasper = data_root / "qasper"
        qasper.mkdir(parents=True)
        papers = []
        for i in range(n):
            pid, qid = f"p{i}", f"q{i}"
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
        (qasper / "dev.jsonl").write_text(
            "\n".join(json.dumps(p) for p in papers) + "\n", encoding="utf-8"
        )
        (qasper / "calibration_pool.jsonl").write_text(
            "\n".join(json.dumps({"paper_id": p["paper_id"],
                                  "question_id": p["qas"][0]["question_id"],
                                  "question": "?"}) for p in papers) + "\n",
            encoding="utf-8",
        )
        return data_root

    def test_completed_cells_are_skipped_in_place(self, tmp_path: Path, monkeypatch):
        data_root = self._make_pool(tmp_path, n=3)

        # Pre-seed the canonical run dir with 2 of 3 run-0 cells done.
        runs_root = tmp_path / "outputs" / "runs"
        run_id = "testrun"
        _write_prior_predictions(
            runs_root / run_id, "flat",
            [
                {"dataset": "qasper", "paper_id": "p0", "question_id": "q0",
                 "run_index": 0, "answer_f1": 0.5, "evidence_f1": 0.0},
                {"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
                 "run_index": 0, "answer_f1": 0.7, "evidence_f1": 0.0},
            ],
        )

        invocations: list[int] = []

        def fake_run_flat(**kwargs):
            invocations.append(kwargs["run_index"])
            return ArchitectureResult(architecture="flat", predicted_answer="z")

        monkeypatch.setattr(cli, "run_flat", fake_run_flat)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

        out = cli.run_dry_run(
            architectures=["flat"],
            datasets=["qasper"],
            answerer_provider="gemini",
            answerer_model="m",
            embedder_model="bge-m3",
            naive_rag_top_k=8,
            data_root=data_root,
            out_dir=tmp_path / "out",
            run_id=run_id,
            run_index=0,
            runs_root=runs_root,
        )

        # Only the missing cell (p2, q2) ran, at run_index 0.
        assert invocations == [0]
        # Verdict counts all 3 (2 prior + 1 new).
        assert out["per_arch_counts"]["flat"] == 3
        # Resume-in-place: same dir, no new run_id minted.
        assert out["run_id"] == run_id
        assert Path(out["predictions_dir"]) == runs_root / run_id
        # Manifest flipped to complete.
        manifest = json.loads(
            (runs_root / run_id / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "complete"

    def test_repeat_run_index_runs_again_in_same_dir(self, tmp_path: Path, monkeypatch):
        """run_index 1 is NOT skipped just because run_index 0 is done;
        the repeat appends into the SAME dir."""
        data_root = self._make_pool(tmp_path, n=2)
        runs_root = tmp_path / "outputs" / "runs"
        run_id = "testrun2"
        # All run-0 cells done.
        _write_prior_predictions(
            runs_root / run_id, "flat",
            [
                {"dataset": "qasper", "paper_id": "p0", "question_id": "q0",
                 "run_index": 0, "answer_f1": 0.5},
                {"dataset": "qasper", "paper_id": "p1", "question_id": "q1",
                 "run_index": 0, "answer_f1": 0.5},
            ],
        )

        seen: list[int] = []

        def fake_run_flat(**kwargs):
            seen.append(kwargs["run_index"])
            return ArchitectureResult(architecture="flat", predicted_answer="z")

        monkeypatch.setattr(cli, "run_flat", fake_run_flat)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

        out = cli.run_dry_run(
            architectures=["flat"],
            datasets=["qasper"],
            answerer_provider="gemini",
            answerer_model="m",
            embedder_model="bge-m3",
            naive_rag_top_k=8,
            data_root=data_root,
            out_dir=tmp_path / "out",
            run_id=run_id,
            run_index=1,
            runs_root=runs_root,
        )

        # Both papers ran at run_index 1 (run_index 0 already done).
        assert seen == [1, 1]
        # 2 prior run-0 rows + 2 new run-1 rows in the same file.
        assert out["per_arch_counts"]["flat"] == 4
