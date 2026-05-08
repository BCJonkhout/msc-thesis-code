"""Tests for pilot.cli.step_4_variance.

The variance helper takes ≥ 2 run dirs, intersects their predictions
JSONL files on (paper_id, question_id), and reports per-run macro F1
plus across-run summaries (mean, SD, SEM, 95% CI half-width, per-
question SD distribution).

Coverage:
  - basic two-run computation against a synthetic pair
  - three-run case exercises stdev across n>2
  - empty intersection raises a clear error
  - missing predictions JSONL raises FileNotFoundError
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot.cli import step_4_variance as v


def _write_predictions(
    run_dir: Path, arch: str, rows: list[dict],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{arch}_predictions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r))
            fh.write("\n")


def _make(qid: str, f1: float) -> dict:
    return {
        "dataset": "qasper",
        "paper_id": "p",
        "question_id": qid,
        "predicted_answer": "x",
        "answer_f1": f1,
        "evidence_f1": 0.0,
    }


class TestComputeVariance:
    def test_two_run_basic(self, tmp_path: Path):
        a = tmp_path / "run_a"
        b = tmp_path / "run_b"
        _write_predictions(a, "flat", [_make("q1", 0.6), _make("q2", 0.4)])
        _write_predictions(b, "flat", [_make("q1", 0.8), _make("q2", 0.2)])

        out = v.compute_variance([a, b], "flat", "answer_f1")
        # macros are (0.6+0.4)/2 = 0.5 and (0.8+0.2)/2 = 0.5
        assert out["per_run_macro"] == [0.5, 0.5]
        assert out["macro_mean"] == 0.5
        assert out["macro_sd"] == 0.0  # both runs identical macros
        # per-question SD: sd([0.6,0.8]) = 0.1414; sd([0.4,0.2]) = 0.1414
        assert abs(out["per_question_sd_mean"] - 0.1414) < 1e-3
        assert out["n_runs"] == 2
        assert out["n_questions_intersected"] == 2

    def test_three_run_with_macro_drift(self, tmp_path: Path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        c = tmp_path / "c"
        _write_predictions(a, "flat", [_make("q1", 0.5), _make("q2", 0.7)])
        _write_predictions(b, "flat", [_make("q1", 0.6), _make("q2", 0.8)])
        _write_predictions(c, "flat", [_make("q1", 0.7), _make("q2", 0.9)])

        out = v.compute_variance([a, b, c], "flat", "answer_f1")
        # macros: 0.6, 0.7, 0.8 → mean 0.7, sd ≈ 0.1
        assert out["macro_mean"] == 0.7
        assert abs(out["macro_sd"] - 0.1) < 1e-3
        # SEM = 0.1 / sqrt(3) ≈ 0.0577
        assert abs(out["sem"] - 0.0577) < 1e-3
        # CI half-width = 1.96 * SEM
        assert abs(out["ci95_half_width"] - 0.1132) < 1e-3
        assert out["n_runs"] == 3

    def test_empty_intersection_raises(self, tmp_path: Path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _write_predictions(a, "flat", [_make("q1", 0.5)])
        _write_predictions(b, "flat", [_make("qZ", 0.5)])
        with pytest.raises(RuntimeError, match="no questions"):
            v.compute_variance([a, b], "flat", "answer_f1")

    def test_missing_predictions_file_raises(self, tmp_path: Path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _write_predictions(a, "flat", [_make("q1", 0.5)])
        b.mkdir()  # b exists but has no predictions
        with pytest.raises(FileNotFoundError, match="missing"):
            v.compute_variance([a, b], "flat", "answer_f1")
