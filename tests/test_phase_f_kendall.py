"""Tests for pilot.cli.phase_f_kendall.

The module is pure offline — takes two run dirs as input and
emits a verdict dict. Tests cover:

  - macro F1 aggregation per architecture from a fake run dir
  - rank ordering ties broken alphabetically
  - Kendall's τ for known cases (identical, reversed, swap-one-pair)
  - decision rule: τ ≥ 0.67 ⇒ STABLE_RANK, else RANK_DEPENDS_ON_ANSWERER
  - error path when fewer than 2 common architectures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot.cli import phase_f_kendall as pf


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _make_run(run_dir: Path, per_arch_f1s: dict[str, list[float]]) -> Path:
    """Fake a run dir with per-arch predictions JSONL files; each row
    is one QASPER question with the given answer_f1."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for arch, f1s in per_arch_f1s.items():
        path = run_dir / f"{arch}_predictions.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for i, f in enumerate(f1s):
                row = {
                    "dataset": "qasper",
                    "paper_id": f"p{i}",
                    "question_id": f"q{i}",
                    "answer_f1": f,
                    "evidence_f1": 0.0,
                }
                fh.write(json.dumps(row))
                fh.write("\n")
    return run_dir


# ──────────────────────────────────────────────────────────────────────
# _macro_f1_per_arch
# ──────────────────────────────────────────────────────────────────────

class TestMacroF1PerArch:
    def test_averages_qasper_f1(self, tmp_path: Path):
        run = _make_run(tmp_path / "run", {
            "flat": [0.5, 0.7, 0.3],
            "naive_rag": [0.4, 0.6, 0.5],
        })
        scores = pf._macro_f1_per_arch(run)
        assert abs(scores["flat"] - 0.5) < 1e-9
        assert abs(scores["naive_rag"] - 0.5) < 1e-9

    def test_skips_archs_with_no_qasper_rows(self, tmp_path: Path):
        run = tmp_path / "run"
        run.mkdir()
        # Only NovelQA rows for flat — should be skipped.
        with (run / "flat_predictions.jsonl").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "dataset": "novelqa", "paper_id": "B01",
                "question_id": "Q1", "predicted_letter": "A",
            }) + "\n")
        scores = pf._macro_f1_per_arch(run)
        assert scores == {}


# ──────────────────────────────────────────────────────────────────────
# _rank_by_f1
# ──────────────────────────────────────────────────────────────────────

class TestRankByF1:
    def test_descending_order(self):
        ranking = pf._rank_by_f1({"a": 0.5, "b": 0.7, "c": 0.3})
        assert ranking == ["b", "a", "c"]

    def test_ties_broken_alphabetically(self):
        # All three at 0.5 → alphabetical.
        ranking = pf._rank_by_f1({"c": 0.5, "a": 0.5, "b": 0.5})
        assert ranking == ["a", "b", "c"]


# ──────────────────────────────────────────────────────────────────────
# _kendall_tau
# ──────────────────────────────────────────────────────────────────────

class TestKendallTau:
    def test_identical_rankings_tau_is_one(self):
        tau, c, d = pf._kendall_tau(["a", "b", "c", "d"], ["a", "b", "c", "d"])
        assert tau == 1.0
        assert d == 0

    def test_reversed_rankings_tau_is_minus_one(self):
        tau, c, d = pf._kendall_tau(["a", "b", "c", "d"], ["d", "c", "b", "a"])
        assert tau == -1.0
        assert c == 0

    def test_swap_one_adjacent_pair(self):
        # 4 architectures → C(4,2) = 6 pairs. Swap (b, c) reverses
        # one ordered pair: 5 concordant, 1 discordant → τ = (5-1)/6 = 2/3.
        tau, c, d = pf._kendall_tau(["a", "b", "c", "d"], ["a", "c", "b", "d"])
        assert abs(tau - 2 / 3) < 1e-9
        assert c == 5 and d == 1

    def test_no_common_items_returns_zero(self):
        tau, c, d = pf._kendall_tau(["a", "b"], ["c", "d"])
        assert tau == 0.0


# ──────────────────────────────────────────────────────────────────────
# compute_phase_f decision rule
# ──────────────────────────────────────────────────────────────────────

class TestComputePhaseF:
    def test_stable_rank_when_tau_above_threshold(self, tmp_path: Path):
        run_a = _make_run(tmp_path / "a", {
            "flat": [0.3], "naive_rag": [0.5], "raptor": [0.7], "graphrag": [0.4],
        })
        run_b = _make_run(tmp_path / "b", {
            "flat": [0.25], "naive_rag": [0.45], "raptor": [0.6], "graphrag": [0.35],
        })
        v = pf.compute_phase_f(run_a, run_b)
        # Rankings identical: raptor > naive_rag > graphrag > flat in both.
        assert v["kendalls_tau"] == 1.0
        assert v["decision"] == "STABLE_RANK"

    def test_rank_depends_when_tau_below_threshold(self, tmp_path: Path):
        run_a = _make_run(tmp_path / "a", {
            "flat": [0.3], "naive_rag": [0.5], "raptor": [0.7], "graphrag": [0.4],
        })
        # Reversed ranking under the alternate answerer.
        run_b = _make_run(tmp_path / "b", {
            "flat": [0.7], "naive_rag": [0.5], "raptor": [0.3], "graphrag": [0.6],
        })
        v = pf.compute_phase_f(run_a, run_b)
        assert v["kendalls_tau"] < 0.67
        assert v["decision"] == "RANK_DEPENDS_ON_ANSWERER"

    def test_error_when_fewer_than_two_common_archs(self, tmp_path: Path):
        run_a = _make_run(tmp_path / "a", {"flat": [0.5]})
        run_b = _make_run(tmp_path / "b", {"naive_rag": [0.4]})
        v = pf.compute_phase_f(run_a, run_b)
        assert "error" in v
