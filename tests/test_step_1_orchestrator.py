"""Step 1 smoke orchestrator: candidate walker, decision rule, exit codes."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pilot.cli import step_1_smoke
from pilot.cli.step_1_smoke import (
    CandidateVerdict,
    TierResult,
    _aggregate_candidate_status,
    _is_malformed,
    _build_smoke_doc,
)
from pilot.providers.base import ProviderResult


# ─── Pure helpers ─────────────────────────────────────────────────────────────
def test_smoke_doc_is_deterministic() -> None:
    a = _build_smoke_doc(5000)
    b = _build_smoke_doc(5000)
    assert a == b
    # Length is approximately tier_tokens * 4 chars.
    assert abs(len(a) - 5000 * 4) <= 4


def test_malformed_detector_catches_thinking_tags() -> None:
    bad, reason = _is_malformed("Sure! <thinking>let me reason</thinking> A")
    assert bad
    assert reason == "leaked_control_tag:<thinking>"


def test_malformed_detector_passes_clean_output() -> None:
    bad, reason = _is_malformed("Paris is the capital of France.")
    assert not bad
    assert reason is None


def test_malformed_detector_catches_empty() -> None:
    bad, reason = _is_malformed("")
    assert bad
    assert reason == "empty_response"


# ─── Decision-rule aggregation ────────────────────────────────────────────────
def _ok_tier(tier: str) -> TierResult:
    return TierResult(
        tier=tier, tier_tokens=10, status="pass",
        wallclock_s=1.0, output_tokens=20, response_first_120="ok",
    )


def test_pass_when_all_three_tiers_pass() -> None:
    tiers = [_ok_tier("5k"), _ok_tier("150k"), _ok_tier("600k")]
    status, reason = _aggregate_candidate_status(tiers)
    assert status == "pass"
    assert reason is None


def test_fail_on_600k_latency_timeout() -> None:
    tiers = [
        _ok_tier("5k"), _ok_tier("150k"),
        TierResult(tier="600k", tier_tokens=600_000, status="fail_latency",
                   wallclock_s=600.0, failure_reason="latency_timeout_600s"),
    ]
    status, reason = _aggregate_candidate_status(tiers)
    assert status == "fail"
    assert reason == "600k_latency_timeout"


def test_fail_on_two_malformed_outputs() -> None:
    tiers = [
        TierResult(tier="5k", tier_tokens=5000, status="fail_malformed",
                   failure_reason="leaked_control_tag:<thinking>"),
        TierResult(tier="150k", tier_tokens=150_000, status="fail_malformed",
                   failure_reason="leaked_control_tag:<thinking>"),
        _ok_tier("600k"),
    ]
    status, reason = _aggregate_candidate_status(tiers)
    assert status == "fail"
    assert reason == "malformed_on_2_of_3_smokes"


def test_pass_with_one_malformed_output() -> None:
    tiers = [
        _ok_tier("5k"),
        TierResult(tier="150k", tier_tokens=150_000, status="fail_malformed",
                   failure_reason="empty_response"),
        _ok_tier("600k"),
    ]
    status, reason = _aggregate_candidate_status(tiers)
    assert status == "pass"  # one is allowed per § 5.8 row #6


def test_pass_with_context_too_small_on_600k() -> None:
    """Candidates with context window < 600k are exempt from that tier."""
    tiers = [
        _ok_tier("5k"), _ok_tier("150k"),
        TierResult(tier="600k", tier_tokens=600_000, status="context_too_small",
                   failure_reason="context_window=256000 < tier_tokens=600000"),
    ]
    status, reason = _aggregate_candidate_status(tiers)
    assert status == "pass"


def test_fail_on_provider_error() -> None:
    tiers = [
        _ok_tier("5k"),
        TierResult(tier="150k", tier_tokens=150_000, status="error",
                   failure_reason="RuntimeError('oops')"),
        _ok_tier("600k"),
    ]
    status, reason = _aggregate_candidate_status(tiers)
    assert status == "fail"
    assert "150k" in reason


# ─── End-to-end orchestrator with mocks ───────────────────────────────────────
def _write_minimal_models_yaml(path: Path) -> None:
    data = {
        "closed_candidates": [
            {"id": "test-closed", "provider": "anthropic",
             "context_window_tokens": 1_000_000,
             "source": "pilot:test"},
            {"id": "test-small", "provider": "anthropic",
             "context_window_tokens": 100_000,
             "source": "pilot:test"},
        ],
        "open_weights_candidates": [
            {"id": "test-open", "provider": "openrouter",
             "context_window_tokens": 1_000_000,
             "source": "pilot:test"},
        ],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _ok_provider_result(text: str = "Paris is the capital.") -> ProviderResult:
    return ProviderResult(
        text=text,
        uncached_input_tokens=1000,
        cached_input_tokens=0,
        output_tokens=20,
        provider_request_id="test-req",
        wallclock_s=0.5,
    )


def test_no_keys_skips_everything(tmp_path: Path) -> None:
    yaml_path = tmp_path / "models.yaml"
    _write_minimal_models_yaml(yaml_path)
    with patch.dict(os.environ, {}, clear=True):
        summary = step_1_smoke.run_step_1(
            models_yaml=yaml_path,
            tiers=["5k"],
            out_dir=tmp_path / "verdicts",
            ledger_root=tmp_path / "runs",
        )
    assert summary["candidates_tested"] == 0
    assert summary["candidates_skipped"] == 3


def test_one_candidate_passes_with_mocked_provider(tmp_path: Path) -> None:
    yaml_path = tmp_path / "models.yaml"
    _write_minimal_models_yaml(yaml_path)
    env = {"ANTHROPIC_API_KEY": "fake"}

    fake_provider = type("FakeProvider", (), {
        "name": "anthropic",
        "call": lambda self, *a, **kw: _ok_provider_result(),
    })()

    with patch.dict(os.environ, env, clear=True):
        with patch.object(step_1_smoke, "get_provider", return_value=fake_provider):
            summary = step_1_smoke.run_step_1(
                models_yaml=yaml_path,
                tiers=["5k", "150k"],
                only_candidates=["test-closed"],
                out_dir=tmp_path / "verdicts",
                ledger_root=tmp_path / "runs",
            )

    assert summary["candidates_tested"] == 1
    assert summary["passed"] == ["test-closed"]
    assert summary["all_passed"] is True


def test_small_context_candidate_skips_600k_tier(tmp_path: Path) -> None:
    yaml_path = tmp_path / "models.yaml"
    _write_minimal_models_yaml(yaml_path)
    env = {"ANTHROPIC_API_KEY": "fake"}

    fake_provider = type("FakeProvider", (), {
        "name": "anthropic",
        "call": lambda self, *a, **kw: _ok_provider_result(),
    })()

    with patch.dict(os.environ, env, clear=True):
        with patch.object(step_1_smoke, "get_provider", return_value=fake_provider):
            summary = step_1_smoke.run_step_1(
                models_yaml=yaml_path,
                tiers=["5k", "150k", "600k"],
                only_candidates=["test-small"],  # 100k context
                out_dir=tmp_path / "verdicts",
                ledger_root=tmp_path / "runs",
            )

    verdict = next(v for v in summary["verdicts"] if v["model"] == "test-small")
    statuses = {t["tier"]: t["status"] for t in verdict["tier_results"]}
    assert statuses["5k"] == "pass"
    assert statuses["150k"] == "context_too_small"
    assert statuses["600k"] == "context_too_small"
    assert verdict["status"] == "pass"  # context-too-small does not fail


def test_malformed_output_counted(tmp_path: Path) -> None:
    yaml_path = tmp_path / "models.yaml"
    _write_minimal_models_yaml(yaml_path)
    env = {"ANTHROPIC_API_KEY": "fake"}

    fake_provider = type("FakeProvider", (), {
        "name": "anthropic",
        "call": lambda self, *a, **kw: _ok_provider_result(
            text="<thinking>hmm</thinking> Paris."
        ),
    })()

    with patch.dict(os.environ, env, clear=True):
        with patch.object(step_1_smoke, "get_provider", return_value=fake_provider):
            summary = step_1_smoke.run_step_1(
                models_yaml=yaml_path,
                tiers=["5k", "150k", "600k"],
                only_candidates=["test-closed"],
                out_dir=tmp_path / "verdicts",
                ledger_root=tmp_path / "runs",
            )

    verdict = next(v for v in summary["verdicts"] if v["model"] == "test-closed")
    statuses = {t["tier"]: t["status"] for t in verdict["tier_results"]}
    assert all(s == "fail_malformed" for s in statuses.values())
    assert verdict["status"] == "fail"
    assert "malformed_on_3_of_3_smokes" == verdict["reason"]
