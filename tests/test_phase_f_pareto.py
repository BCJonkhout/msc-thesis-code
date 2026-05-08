"""Tests for pilot.cli.phase_f_pareto.

The Pareto aggregator combines per-candidate F1 (from predictions
JSONL) with per-candidate cost (from ledger.jsonl) and computes
Kendall's τ vs a reference run dir. Coverage:

  - 4-arch reference + 2-candidate sweep produces a sorted table
    (cheapest first)
  - τ = +1.0 case (all ranks identical) → STABLE_RANK
  - τ = -1.0 case (one swap on 2 archs) → RANK_DEPENDS_ON_ANSWERER
  - missing predictions raises a sane error per candidate, not the
    whole run
  - cost aggregation honours run_index = 0 only
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pilot.cli import phase_f_pareto as pareto


def _write_predictions(run_dir: Path, arch: str, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{arch}_predictions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r))
            fh.write("\n")


def _write_ledger(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "ledger.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            full = {
                "timestamp": "2026-05-08T12:00:00",
                "architecture": "flat",
                "stage": "generate",
                "model": "test-model",
                "run_index": 0,
                "uncached_input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "wallclock_s": 0.1,
                "gpu_s_estimate": 0.0,
                "provider_request_id": "req",
                "prompt_hash": "h",
                "response_hash": "h",
                "seed": None,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": None,
                "provider_region": None,
                "failed": False,
                "failure_reason": None,
                **r,
            }
            fh.write(json.dumps(full))
            fh.write("\n")


def _qa(qid: str, f1: float, dataset: str = "qasper") -> dict:
    return {
        "dataset": dataset, "paper_id": "p", "question_id": qid,
        "predicted_answer": "x", "answer_f1": f1,
        "evidence_f1": 0.0,
    }


def _make_4arch_run(run_dir: Path, scores: dict[str, float]) -> None:
    """Write per-arch predictions where every QASPER question gets the
    given macro F1 (so macro = each row's f1)."""
    for arch, f1 in scores.items():
        _write_predictions(run_dir, arch, [_qa("q1", f1), _qa("q2", f1)])


@pytest.fixture
def price_card() -> dict[str, Any]:
    return {
        "providers": {
            "test": {
                "models": {
                    "test-model": {
                        "input_uncached": 1.0,
                        "input_cached_read": 0.1,
                        "output": 5.0,
                    },
                },
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Stable-rank (τ = +1.0) → STABLE_RANK
# ──────────────────────────────────────────────────────────────────────

class TestStableRank:
    def test_identical_rankings_yield_tau_one(self, tmp_path, price_card):
        ref = tmp_path / "ref"
        cand = tmp_path / "cand"
        _make_4arch_run(ref, {"flat": 0.34, "naive_rag": 0.37, "raptor": 0.27, "graphrag": 0.05})
        _make_4arch_run(cand, {"flat": 0.36, "naive_rag": 0.42, "raptor": 0.26, "graphrag": 0.03})
        _write_ledger(cand, [{"uncached_input_tokens": 100_000, "output_tokens": 100}])

        verdict = pareto.build_pareto_table(
            [cand], ref, reference_label="ref", price_card=price_card,
        )
        assert verdict["candidates"][0]["kendalls_tau_vs_reference"] == 1.0
        assert verdict["candidates"][0]["decision"] == "STABLE_RANK"
        assert verdict["candidates"][0]["total_usd"] > 0


# ──────────────────────────────────────────────────────────────────────
# Rank-inversion (τ = -1.0 over 2 archs) → RANK_DEPENDS_ON_ANSWERER
# ──────────────────────────────────────────────────────────────────────

class TestRankInversion:
    def test_two_arch_inversion_yields_tau_minus_one(self, tmp_path, price_card):
        ref = tmp_path / "ref"
        cand = tmp_path / "cand"
        _make_4arch_run(ref, {"flat": 0.34, "naive_rag": 0.37})
        _make_4arch_run(cand, {"flat": 0.43, "naive_rag": 0.24})
        _write_ledger(cand, [{"uncached_input_tokens": 50_000, "output_tokens": 50}])

        verdict = pareto.build_pareto_table(
            [cand], ref, reference_label="ref", price_card=price_card,
        )
        c = verdict["candidates"][0]
        assert c["kendalls_tau_vs_reference"] == -1.0
        assert c["decision"] == "RANK_DEPENDS_ON_ANSWERER"


# ──────────────────────────────────────────────────────────────────────
# Cheapest-first sort + per-candidate error isolation
# ──────────────────────────────────────────────────────────────────────

class TestSortAndErrorIsolation:
    def test_table_sorted_by_cost_ascending(self, tmp_path, price_card):
        ref = tmp_path / "ref"
        cheap = tmp_path / "cheap"
        pricy = tmp_path / "pricy"
        _make_4arch_run(ref, {"flat": 0.3, "naive_rag": 0.4})
        _make_4arch_run(cheap, {"flat": 0.3, "naive_rag": 0.4})
        _make_4arch_run(pricy, {"flat": 0.3, "naive_rag": 0.4})
        _write_ledger(cheap, [{"uncached_input_tokens": 10_000}])
        _write_ledger(pricy, [{"uncached_input_tokens": 1_000_000}])

        verdict = pareto.build_pareto_table(
            [pricy, cheap], ref, reference_label="ref", price_card=price_card,
        )
        # First entry should be cheap (cost ascending)
        labels_in_order = [c["run_dir"] for c in verdict["candidates"]]
        assert labels_in_order[0] == str(cheap)
        assert labels_in_order[1] == str(pricy)

    def test_missing_predictions_isolated_to_one_row(self, tmp_path, price_card):
        ref = tmp_path / "ref"
        good = tmp_path / "good"
        bad = tmp_path / "bad"
        _make_4arch_run(ref, {"flat": 0.3, "naive_rag": 0.4})
        _make_4arch_run(good, {"flat": 0.3, "naive_rag": 0.4})
        _write_ledger(good, [{"uncached_input_tokens": 10_000}])
        bad.mkdir()  # empty: no predictions, no ledger

        verdict = pareto.build_pareto_table(
            [good, bad], ref, reference_label="ref", price_card=price_card,
        )
        # bad row carries an error key; good row computes normally
        rows_by_dir = {r["run_dir"]: r for r in verdict["candidates"]}
        assert "error" in rows_by_dir[str(bad)]
        assert "error" not in rows_by_dir[str(good)]


# ──────────────────────────────────────────────────────────────────────
# Cost-attribution honours run_index = 0 only
# ──────────────────────────────────────────────────────────────────────

class TestCostAttribution:
    def test_run_index_nonzero_excluded(self, tmp_path, price_card):
        ref = tmp_path / "ref"
        cand = tmp_path / "cand"
        _make_4arch_run(ref, {"flat": 0.3, "naive_rag": 0.4})
        _make_4arch_run(cand, {"flat": 0.3, "naive_rag": 0.4})
        _write_ledger(cand, [
            {"run_index": 0, "uncached_input_tokens": 1_000_000},
            {"run_index": 1, "uncached_input_tokens": 1_000_000},
            {"run_index": 2, "uncached_input_tokens": 1_000_000},
        ])

        verdict = pareto.build_pareto_table(
            [cand], ref, reference_label="ref", price_card=price_card,
        )
        # Only run_index=0 row should contribute; total ≈ $1.00
        assert abs(verdict["candidates"][0]["total_usd"] - 1.0) < 1e-3

    def test_failed_rows_excluded(self, tmp_path, price_card):
        ref = tmp_path / "ref"
        cand = tmp_path / "cand"
        _make_4arch_run(ref, {"flat": 0.3, "naive_rag": 0.4})
        _make_4arch_run(cand, {"flat": 0.3, "naive_rag": 0.4})
        _write_ledger(cand, [
            {"failed": False, "uncached_input_tokens": 500_000},
            {"failed": True, "uncached_input_tokens": 500_000,
             "failure_reason": "timeout"},
        ])
        verdict = pareto.build_pareto_table(
            [cand], ref, reference_label="ref", price_card=price_card,
        )
        # Only the non-failed row contributes ($0.50)
        assert abs(verdict["candidates"][0]["total_usd"] - 0.5) < 1e-3
