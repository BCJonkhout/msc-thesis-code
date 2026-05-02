"""Cost ledger: roundtrip JSONL writes and price-card sums."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pilot.ledger import CostLedger, Stage, sha256_hex
from pilot.price_card import compute


@pytest.fixture
def price_card() -> dict[str, Any]:
    """Minimal hand-rolled price card for the test."""
    return {
        "providers": {
            "test_provider": {
                "models": {
                    "test-model-1": {
                        "input_uncached": 1.00,   # $1 / 1M tok
                        "input_cached_read": 0.10,
                        "output": 5.00,
                    }
                }
            }
        }
    }


def test_log_call_writes_jsonl(tmp_path: Path) -> None:
    """One call writes one row with the expected fields."""
    ledger = CostLedger(run_id="t1", root=tmp_path)
    with ledger.log_call(
        architecture="flat",
        stage=Stage.GENERATE,
        model="test-model-1",
        prompt="hello world",
    ) as rec:
        rec.uncached_input_tokens = 100
        rec.cached_input_tokens = 50
        rec.output_tokens = 30
        rec.provider_request_id = "req_abc"
        rec.response_hash = sha256_hex("response text")

    rows = ledger.read()
    assert len(rows) == 1
    r = rows[0]
    assert r.architecture == "flat"
    assert r.stage == "generate"
    assert r.model == "test-model-1"
    assert r.uncached_input_tokens == 100
    assert r.cached_input_tokens == 50
    assert r.output_tokens == 30
    assert r.provider_request_id == "req_abc"
    assert r.prompt_hash == sha256_hex("hello world")
    assert r.timestamp  # filled by ledger
    assert r.wallclock_s >= 0


def test_log_call_handles_exception(tmp_path: Path) -> None:
    """An exception in the context still writes a `failed=true` row."""
    ledger = CostLedger(run_id="t2", root=tmp_path)
    with pytest.raises(ValueError):
        with ledger.log_call(
            architecture="flat",
            stage=Stage.GENERATE,
            model="test-model-1",
            prompt="boom",
        ) as rec:
            rec.uncached_input_tokens = 5
            raise ValueError("simulated failure")

    rows = ledger.read()
    assert len(rows) == 1
    assert rows[0].failed is True
    assert rows[0].failure_reason and "simulated failure" in rows[0].failure_reason


def test_price_card_sums_run_index_zero_only(tmp_path: Path, price_card: dict) -> None:
    """Option A: only run_index == 0 contributes to the total."""
    ledger = CostLedger(run_id="t3", root=tmp_path)

    # 2 calls at run_index=0 (counted), 2 at run_index=1 (excluded).
    for ridx in (0, 0, 1, 1):
        with ledger.log_call(
            architecture="flat",
            stage=Stage.GENERATE,
            model="test-model-1",
            prompt=f"call {ridx}",
            run_index=ridx,
        ) as rec:
            rec.uncached_input_tokens = 1_000_000   # 1M tokens
            rec.cached_input_tokens = 0
            rec.output_tokens = 100_000             # 100k tokens

    # Each row at $1/1M input + $5/1M output = $1.00 + $0.50 = $1.50.
    # Two run_index=0 rows → $3.00. Run_index=1 rows excluded.
    total = compute(ledger.path, price_card)
    assert total == pytest.approx(3.00, abs=1e-6)


def test_price_card_back_of_envelope(tmp_path: Path, price_card: dict) -> None:
    """Hand-computed total matches the resolver output within $0.01."""
    ledger = CostLedger(run_id="t4", root=tmp_path)

    # Five synthetic rows; all run_index=0; mix of cached/uncached.
    rows_to_write = [
        # (uncached, cached, output)
        (500_000, 0, 50_000),         # $0.50 + $0 + $0.25 = $0.75
        (1_000_000, 0, 100_000),      # $1.00 + $0 + $0.50 = $1.50
        (200_000, 800_000, 20_000),   # $0.20 + $0.08 + $0.10 = $0.38
        (1_000_000, 1_000_000, 200_000),  # $1.00 + $0.10 + $1.00 = $2.10
        (50_000, 0, 5_000),           # $0.05 + $0 + $0.025 = $0.075
    ]
    expected = 0.75 + 1.50 + 0.38 + 2.10 + 0.075  # = 4.805

    for u, c, o in rows_to_write:
        with ledger.log_call(
            architecture="flat",
            stage=Stage.GENERATE,
            model="test-model-1",
            prompt="x",
        ) as rec:
            rec.uncached_input_tokens = u
            rec.cached_input_tokens = c
            rec.output_tokens = o

    total = compute(ledger.path, price_card)
    assert total == pytest.approx(expected, abs=0.01)
