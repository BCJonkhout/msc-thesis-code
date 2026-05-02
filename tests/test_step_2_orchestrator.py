"""Step 2 orchestrator: skip-if-no-key, aggregate verdicts, exit codes."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from pilot.cli import step_2_kvcache
from pilot.providers import CacheControl
from pilot.sanity.kvcache_check import CacheCheckResult


def _fake_pass_result(provider: str, model: str) -> CacheCheckResult:
    return CacheCheckResult(
        provider=provider,
        model=model,
        cache_control="ephemeral_5min",
        doc_tokens_estimate=100_000,
        first_call_wallclock_s=10.0,
        first_call_uncached_tokens=100_000,
        first_call_cached_tokens=0,
        second_call_wallclock_s=2.0,    # 0.2x first call -> latency_pass=True
        second_call_uncached_tokens=5_000,
        second_call_cached_tokens=95_000,
        latency_ratio=0.2,
        latency_pass=True,
        cached_token_ratio=0.95,
        cached_token_pass=True,
        overall_pass=True,
    )


def _fake_fail_result(provider: str, model: str) -> CacheCheckResult:
    return CacheCheckResult(
        provider=provider,
        model=model,
        cache_control="ephemeral_5min",
        doc_tokens_estimate=100_000,
        first_call_wallclock_s=10.0,
        first_call_uncached_tokens=100_000,
        first_call_cached_tokens=0,
        second_call_wallclock_s=9.5,
        second_call_uncached_tokens=100_000,
        second_call_cached_tokens=0,
        latency_ratio=0.95,
        latency_pass=False,
        cached_token_ratio=0.0,
        cached_token_pass=False,
        overall_pass=False,
    )


def test_no_keys_returns_all_skipped(tmp_path: Path) -> None:
    """No env keys present → every provider is `skipped`, nothing tested."""
    with patch.dict(os.environ, {}, clear=True):
        summary = step_2_kvcache.run_step_2(
            doc_tokens=1000,
            cache_control=CacheControl.EPHEMERAL_5MIN,
            out_dir=tmp_path,
        )

    assert summary["providers_tested"] == []
    assert set(summary["providers_skipped"]) == {
        "anthropic", "openai", "gemini", "dashscope", "openrouter", "xai"
    }
    assert summary["all_passed"] is False
    assert summary["any_passed"] is False
    # Verdict file must still be written for the audit trail.
    out_files = list(tmp_path.glob("step_2_kvcache_*.json"))
    assert len(out_files) == 1
    written = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert written["providers_tested"] == []


def test_one_provider_passes(tmp_path: Path) -> None:
    """One provider has a key and passes; others are skipped."""
    env = {"ANTHROPIC_API_KEY": "fake-key"}
    with patch.dict(os.environ, env, clear=True):
        with patch.object(
            step_2_kvcache,
            "run_cache_check",
            return_value=_fake_pass_result("anthropic", "claude-sonnet-4-6-20260217"),
        ):
            summary = step_2_kvcache.run_step_2(
                doc_tokens=1000,
                cache_control=CacheControl.EPHEMERAL_5MIN,
                out_dir=tmp_path,
            )

    assert summary["providers_tested"] == ["anthropic"]
    assert summary["all_passed"] is True
    anthropic_entry = next(p for p in summary["per_provider"] if p["provider"] == "anthropic")
    assert anthropic_entry["status"] == "pass"
    assert anthropic_entry["result"]["overall_pass"] is True


def test_one_passes_one_fails(tmp_path: Path) -> None:
    """Mixed verdict: one provider passes, one fails. all_passed=False, any_passed=True."""
    env = {"ANTHROPIC_API_KEY": "fake-a", "OPENAI_API_KEY": "fake-b"}

    def fake_check(*, provider_name: str, model: str, **_kwargs):
        if provider_name == "anthropic":
            return _fake_pass_result(provider_name, model)
        return _fake_fail_result(provider_name, model)

    with patch.dict(os.environ, env, clear=True):
        with patch.object(step_2_kvcache, "run_cache_check", side_effect=fake_check):
            summary = step_2_kvcache.run_step_2(
                doc_tokens=1000,
                cache_control=CacheControl.EPHEMERAL_5MIN,
                out_dir=tmp_path,
            )

    assert set(summary["providers_tested"]) == {"anthropic", "openai"}
    assert summary["all_passed"] is False
    assert summary["any_passed"] is True
    statuses = {p["provider"]: p["status"] for p in summary["per_provider"] if "status" in p}
    assert statuses["anthropic"] == "pass"
    assert statuses["openai"] == "fail"


def test_provider_exception_recorded_as_error(tmp_path: Path) -> None:
    """If the underlying check raises, we record `error` rather than crashing."""
    env = {"DASHSCOPE_API_KEY": "fake-d"}
    with patch.dict(os.environ, env, clear=True):
        with patch.object(
            step_2_kvcache,
            "run_cache_check",
            side_effect=RuntimeError("network unreachable"),
        ):
            summary = step_2_kvcache.run_step_2(
                doc_tokens=1000,
                cache_control=CacheControl.EPHEMERAL_5MIN,
                out_dir=tmp_path,
            )

    dashscope_entry = next(p for p in summary["per_provider"] if p["provider"] == "dashscope")
    assert dashscope_entry["status"] == "error"
    assert "network unreachable" in dashscope_entry["reason"]
    assert summary["all_passed"] is False


def test_provider_subset_filter(tmp_path: Path) -> None:
    """`providers=` argument restricts which adapters are walked."""
    env = {"ANTHROPIC_API_KEY": "fake", "OPENAI_API_KEY": "fake"}
    with patch.dict(os.environ, env, clear=True):
        with patch.object(
            step_2_kvcache,
            "run_cache_check",
            return_value=_fake_pass_result("anthropic", "claude-sonnet-4-6-20260217"),
        ):
            summary = step_2_kvcache.run_step_2(
                providers=["anthropic"],
                doc_tokens=1000,
                cache_control=CacheControl.EPHEMERAL_5MIN,
                out_dir=tmp_path,
            )

    assert summary["providers_requested"] == ["anthropic"]
    assert {p["provider"] for p in summary["per_provider"]} == {"anthropic"}
