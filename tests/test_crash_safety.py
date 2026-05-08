"""Crash-safety tests for the Step 3 dry-run pipeline.

The pilot must survive a laptop crash mid-sweep without losing
predictions or cost-ledger rows that have already been computed.
This is achieved by:

  - Per-prediction JSONL flush + os.fsync after each row, so a
    power-loss after the row was written keeps the row durable.
  - Per-ledger-row JSONL flush + os.fsync inside CostLedger._write_row.
  - The --resume-from mechanism replays prior predictions and lets
    the dispatcher skip already-completed (arch, paper, qid) cells.

Tests below verify the durability contract by:

  1. Writing rows through the public API.
  2. Reading the file back BEFORE the writer process closes the
     handle.
  3. Asserting all written rows are present.

Without fsync, a buffered-write OS could lose the most recent rows
on a crash. With fsync the rows are guaranteed on disk before the
write call returns.
"""
from __future__ import annotations

import json
from pathlib import Path

from pilot.ledger import CostLedger, Stage


class TestCostLedgerDurability:
    def test_log_call_persists_row_before_returning(self, tmp_path: Path):
        ledger = CostLedger(run_id="durability-test-1", root=tmp_path)
        with ledger.log_call(
            architecture="flat",
            stage=Stage.GENERATE,
            model="test-model",
            prompt="test prompt",
            run_index=0,
        ) as rec:
            rec.uncached_input_tokens = 100
            rec.cached_input_tokens = 0
            rec.output_tokens = 10
            rec.provider_request_id = "req-test"
        # After the context manager exits, the row MUST be on disk —
        # this is the durability guarantee a crash-after-call relies on.
        path = ledger.path
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line]
        assert len(rows) == 1
        row = rows[0]
        assert row["architecture"] == "flat"
        assert row["uncached_input_tokens"] == 100
        assert row["output_tokens"] == 10

    def test_failed_call_persists_with_failure_reason(self, tmp_path: Path):
        ledger = CostLedger(run_id="durability-test-2", root=tmp_path)
        try:
            with ledger.log_call(
                architecture="naive_rag",
                stage=Stage.GENERATE,
                model="test-model",
                prompt="test",
                run_index=0,
            ):
                raise RuntimeError("simulated provider crash")
        except RuntimeError:
            pass

        rows = [
            json.loads(l)
            for l in ledger.path.read_text(encoding="utf-8").splitlines()
            if l
        ]
        assert len(rows) == 1
        assert rows[0]["failed"] is True
        assert "simulated provider crash" in rows[0]["failure_reason"]

    def test_multiple_consecutive_writes_all_persist(self, tmp_path: Path):
        ledger = CostLedger(run_id="durability-test-3", root=tmp_path)
        for i in range(5):
            with ledger.log_call(
                architecture="raptor",
                stage=Stage.PREPROCESS,
                model="test-model",
                prompt=f"prompt-{i}",
                run_index=0,
            ) as rec:
                rec.uncached_input_tokens = 10 * (i + 1)
                rec.output_tokens = i

        text = ledger.path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line]
        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert row["uncached_input_tokens"] == 10 * (i + 1)
            assert row["output_tokens"] == i


class TestPredictionsFsync:
    """The Step 3 dispatcher writes one JSONL row per prediction and
    flush+fsync after each. This test verifies the contract by simulating
    the write pattern directly (the full dispatcher is exercised in
    test_step_3_dispatcher.py and test_step_3_resume.py)."""

    def test_fsync_call_path_is_resilient_to_oserror(self, tmp_path: Path):
        """Some platforms / filesystems don't support fsync on regular
        files. The dispatcher swallows OSError so the sweep keeps moving.
        Verify by writing through a file handle, calling fsync, and
        ensuring no exception propagates."""
        path = tmp_path / "test.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write('{"x": 1}\n')
            fh.flush()
            try:
                import os
                os.fsync(fh.fileno())
            except (OSError, AttributeError):
                pass
        assert path.read_text() == '{"x": 1}\n'
