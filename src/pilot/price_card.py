"""Price-card resolver: convert a JSONL ledger into a USD total.

Per pilot plan § 4.1 and § 5.8 row #13 (Option A cost-attribution rule),
deployment cost is computed only from rows with `run_index == 0`. Rows
with `run_index >= 1` exist for variance reporting; they don't add to
the total.

The price card YAML drives the lookup. For each ledger row:
    cost = (uncached_input_tokens * input_uncached
            + cached_input_tokens * input_cached_read
            + output_tokens * output) / 1_000_000

Provider-specific rate selection (e.g., Gemini's above-200k tier,
GPT-5.4's above-272k tier) is handled by the per-row prompt token
count + the model's tier rules. For Step 0 we use the simple tier
(below the boundary) and skip tiered selection until needed.

Storage cost (C_store) is computed separately as a back-of-envelope
per-architecture footprint × storage_horizon_days × storage_rate.
See `compute_storage_cost`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pilot.ledger import CallRecord


def load_price_card(path: Path) -> dict[str, Any]:
    """Load a price card YAML file. Provenance validation is the caller's job."""
    from pilot.provenance import load_config
    return load_config(path)


def _provider_for_model(price_card: dict[str, Any], model_id: str) -> tuple[str, dict[str, Any]] | None:
    """Find which provider hosts a given model_id; return (provider, model_rates)."""
    providers = price_card.get("providers", {})
    for provider_name, provider_block in providers.items():
        models = provider_block.get("models", {})
        if model_id in models:
            return provider_name, models[model_id]
    return None


def _row_cost_usd(row: CallRecord, price_card: dict[str, Any]) -> float:
    """Compute the USD cost of a single ledger row."""
    if row.failed:
        return 0.0
    found = _provider_for_model(price_card, row.model)
    if found is None:
        return 0.0
    _provider, rates = found

    # Simple rate selection: prefer flat rates, fall back to below-tier rates.
    in_uncached = (
        rates.get("input_uncached")
        or rates.get("input_uncached_below_200k")
        or rates.get("input_uncached_below_272k")
        or 0.0
    )
    in_cached = rates.get("input_cached_read", 0.0)
    out = (
        rates.get("output")
        or rates.get("output_below_200k")
        or rates.get("output_below_272k")
        or 0.0
    )

    return (
        row.uncached_input_tokens * in_uncached
        + row.cached_input_tokens * in_cached
        + row.output_tokens * out
    ) / 1_000_000


def compute(ledger_path: Path, price_card: dict[str, Any]) -> float:
    """Sum USD cost across all `run_index == 0` rows of a ledger file.

    Per Option A, runs 2..N are excluded from the total — they exist
    only for variance reporting.
    """
    if not ledger_path.exists():
        return 0.0
    total = 0.0
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("run_index", 0) != 0:
                continue
            row = CallRecord(**{k: d.get(k) for k in CallRecord.__dataclass_fields__})
            total += _row_cost_usd(row, price_card)
    return round(total, 6)


def compute_storage_cost(
    *,
    architecture_footprint_gib: float,
    price_card: dict[str, Any],
    documents: int = 1,
) -> float:
    """Back-of-envelope storage cost over the study horizon.

    architecture_footprint_gib: per-document persistent-artifact size
                                 (e.g., RAPTOR ~0.012 GiB, GraphRAG ~0.1 GiB).
    documents:                   number of distinct documents in the study.
    """
    storage = price_card.get("storage", {})
    rate_block = storage.get("rate_usd_per_gib_month", {})
    horizon_block = storage.get("study_horizon_days", {})
    rate_usd_per_gib_month = (
        rate_block["value"] if isinstance(rate_block, dict) else rate_block
    )
    horizon_days = (
        horizon_block["value"] if isinstance(horizon_block, dict) else horizon_block
    )
    months = horizon_days / 30.0
    return round(
        architecture_footprint_gib * documents * rate_usd_per_gib_month * months,
        6,
    )
