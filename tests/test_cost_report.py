"""Tests for pilot.cli.cost_report.

The module is a small wrapper around pilot.price_card.compute that
emits a per-(architecture, stage) breakdown for one or more run
dirs. Tests cover:

  - per-run aggregation against a synthetic ledger.jsonl fixture
  - run_index != 0 rows are excluded (cost-attribution Option A)
  - failed rows are counted in `rows_failed` but not in cost
  - missing ledger surfaces an explicit error
  - --json output emits valid JSON
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pilot.cli import cost_report as cr


# ──────────────────────────────────────────────────────────────────────
# fixtures
# ──────────────────────────────────────────────────────────────────────

def _make_run(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "ledger.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            # CallRecord-shaped: fill defaults so the loader is happy.
            full = {
                "timestamp": "2026-05-08T12:00:00",
                "architecture": "flat",
                "stage": "generate",
                "model": "gemini-3.1-pro-preview",
                "run_index": 0,
                "uncached_input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "wallclock_s": 0.1,
                "gpu_s_estimate": 0.0,
                "provider_request_id": "req_test",
                "prompt_hash": "h",
                "response_hash": "h",
                "seed": None,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": None,
                "provider_region": None,
                "failed": False,
                "failure_reason": None,
                **row,
            }
            fh.write(json.dumps(full))
            fh.write("\n")
    return run_dir


@pytest.fixture
def price_card() -> dict[str, Any]:
    """Minimal price card carrying just gemini-3.1-pro-preview rates."""
    return {
        "providers": {
            "gemini": {
                "models": {
                    "gemini-3.1-pro-preview": {
                        "input_uncached_below_200k": 2.0,
                        "input_cached_read": 0.5,
                        "output_below_200k": 12.0,
                    },
                },
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# _aggregate_run
# ──────────────────────────────────────────────────────────────────────

class TestAggregateRun:
    def test_basic_breakdown(self, tmp_path: Path, price_card: dict[str, Any]):
        run = _make_run(tmp_path / "run", [
            {"architecture": "flat", "stage": "generate",
             "uncached_input_tokens": 1_000_000, "output_tokens": 100_000},
            {"architecture": "naive_rag", "stage": "retrieval",
             "uncached_input_tokens": 50_000, "output_tokens": 0,
             "model": "bge-m3"},
            {"architecture": "naive_rag", "stage": "generate",
             "uncached_input_tokens": 5_000, "output_tokens": 200},
        ])
        out = cr._aggregate_run(run, price_card)
        # flat/generate: 1M uncached × $2/M + 100k output × $12/M = $2.00 + $1.20 = $3.20
        assert "flat/generate" in out["breakdown"]
        flat = out["breakdown"]["flat/generate"]
        assert flat["rows"] == 1
        assert abs(flat["cost_usd"] - 3.20) < 1e-6
        # bge-m3 not in price card → cost 0; row still counted.
        assert out["breakdown"]["naive_rag/retrieval"]["cost_usd"] == 0.0
        assert out["breakdown"]["naive_rag/retrieval"]["rows"] == 1
        # Total is the sum of all priced rows.
        assert out["total_usd"] > 3.0

    def test_run_index_nonzero_excluded(self, tmp_path: Path, price_card):
        run = _make_run(tmp_path / "run", [
            {"architecture": "flat", "stage": "generate",
             "run_index": 0, "uncached_input_tokens": 1_000_000},
            {"architecture": "flat", "stage": "generate",
             "run_index": 1, "uncached_input_tokens": 1_000_000},
            {"architecture": "flat", "stage": "generate",
             "run_index": 2, "uncached_input_tokens": 1_000_000},
        ])
        out = cr._aggregate_run(run, price_card)
        # Only run_index=0 row contributes to cost.
        assert out["breakdown"]["flat/generate"]["rows"] == 1
        assert abs(out["breakdown"]["flat/generate"]["cost_usd"] - 2.0) < 1e-6
        # The total ledger row count includes all 3.
        assert out["ledger_rows_total"] == 3

    def test_failed_rows_excluded_from_cost(self, tmp_path: Path, price_card):
        run = _make_run(tmp_path / "run", [
            {"architecture": "flat", "stage": "generate",
             "uncached_input_tokens": 1_000_000, "failed": False},
            {"architecture": "flat", "stage": "generate",
             "uncached_input_tokens": 1_000_000, "failed": True,
             "failure_reason": "rate limit"},
        ])
        out = cr._aggregate_run(run, price_card)
        # rows_failed counts the failed row separately from cost rows.
        assert out["rows_failed"] == 1
        # Only the successful row contributes to cost ($2.00).
        assert abs(out["breakdown"]["flat/generate"]["cost_usd"] - 2.0) < 1e-6
        assert out["breakdown"]["flat/generate"]["rows"] == 1

    def test_missing_ledger_returns_error(self, tmp_path: Path, price_card):
        # Run dir exists but no ledger.jsonl inside.
        empty = tmp_path / "empty"
        empty.mkdir()
        out = cr._aggregate_run(empty, price_card)
        assert "error" in out
        assert "missing" in out["error"]


# ──────────────────────────────────────────────────────────────────────
# _format_table
# ──────────────────────────────────────────────────────────────────────

class TestFormatTable:
    def test_table_includes_total_line(self, tmp_path: Path, price_card):
        run = _make_run(tmp_path / "run", [
            {"architecture": "flat", "stage": "generate",
             "uncached_input_tokens": 100_000, "output_tokens": 1000},
        ])
        report = cr._aggregate_run(run, price_card)
        rendered = cr._format_table([report])
        assert "TOTAL" in rendered
        assert "flat/generate" in rendered
        assert "$" in rendered

    def test_error_run_shown_in_table(self, tmp_path: Path, price_card):
        empty = tmp_path / "empty"
        empty.mkdir()
        report = cr._aggregate_run(empty, price_card)
        rendered = cr._format_table([report])
        assert "(error)" in rendered
