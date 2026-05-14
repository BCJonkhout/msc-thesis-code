"""Tests for the Gemini provider's retry-on-transient-error wrapper.

Phase G NovelQA runs three answerer lanes concurrently and pins the
RAPTOR/GraphRAG summary stage on Gemini Flash Lite for the non-Google
lanes. Under that load Google returns 429 (rate limit) and 503
(transient capacity); the SDK's built-in retry is not enough to keep
GraphRAG entity extraction (~350 sequential calls per novel) from
intermittently failing mid-sweep, which leaves the architecture's
preprocessing state half-built and forces a full rebuild on the next
question — exactly what the per-paper preprocessing cache is supposed
to prevent.

Coverage:
  - 503 / 429 / 500 / 502 / 504 are retried with exponential backoff.
  - Non-retryable errors (e.g. 400 ClientError, 401, arbitrary
    exceptions) bubble up immediately without sleeping.
  - The retry succeeds when an intermediate attempt returns normally.
  - All 5 attempts failing → the last exception is raised.
  - Backoff sleeps are bounded and increase across attempts.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pilot.providers.gemini_provider import (
    GeminiProvider,
    _RETRY_ATTEMPTS,
    _RETRYABLE_CODES,
)


class _FakeAPIError(Exception):
    """Stand-in for google.genai.errors.ServerError / ClientError.

    The real SDK errors carry a ``.code`` integer; the retry wrapper
    only inspects that attribute, so the test double doesn't need to
    inherit from the real class.
    """
    def __init__(self, code: int, message: str = "") -> None:
        super().__init__(message or f"code={code}")
        self.code = code


@pytest.fixture
def provider():
    return GeminiProvider(api_key="test-key")


class TestRetryablePredicate:
    @pytest.mark.parametrize("code", sorted(_RETRYABLE_CODES))
    def test_retryable_codes(self, provider, code):
        assert provider._retryable(_FakeAPIError(code)) is True

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 410])
    def test_non_retryable_4xx(self, provider, code):
        assert provider._retryable(_FakeAPIError(code)) is False

    def test_exception_without_code_is_not_retryable(self, provider):
        assert provider._retryable(ValueError("nope")) is False
        assert provider._retryable(RuntimeError()) is False


class TestRetryLoop:
    def _drive(self, provider, side_effects, monkeypatch):
        """Run the retry loop with ``side_effects`` as the mock's
        per-call return/raise sequence. Skips real sleeping."""
        client = MagicMock()
        client.models.generate_content.side_effect = side_effects
        slept: list[float] = []
        monkeypatch.setattr(
            "pilot.providers.gemini_provider.time.sleep",
            lambda s: slept.append(s),
        )
        return (
            provider._generate_with_retry(
                client=client, model="gemini-test",
                prompt="hi", config=None,
            ),
            client.models.generate_content.call_count,
            slept,
        )

    def test_succeeds_on_first_attempt_without_sleeping(
        self, provider, monkeypatch
    ):
        good = MagicMock(name="response")
        result, attempts, slept = self._drive(provider, [good], monkeypatch)
        assert result is good
        assert attempts == 1
        assert slept == []

    def test_retries_503_then_succeeds(self, provider, monkeypatch):
        good = MagicMock(name="response")
        result, attempts, slept = self._drive(
            provider,
            [_FakeAPIError(503), _FakeAPIError(503), good],
            monkeypatch,
        )
        assert result is good
        assert attempts == 3
        assert len(slept) == 2
        # Backoff base is 1s with up to +/-25% jitter -> next attempt ~2s
        assert 0.5 < slept[0] < 2.0
        assert 1.0 < slept[1] < 4.0
        # Strictly increasing on average across attempts
        assert slept[1] > slept[0] * 0.75

    def test_retries_429_RESOURCE_EXHAUSTED(self, provider, monkeypatch):
        good = MagicMock(name="response")
        result, attempts, _ = self._drive(
            provider, [_FakeAPIError(429), good], monkeypatch,
        )
        assert result is good
        assert attempts == 2

    def test_non_retryable_400_does_not_retry(self, provider, monkeypatch):
        with pytest.raises(_FakeAPIError) as excinfo:
            self._drive(provider, [_FakeAPIError(400)], monkeypatch)
        assert excinfo.value.code == 400

    def test_arbitrary_exception_does_not_retry(self, provider, monkeypatch):
        with pytest.raises(ValueError, match="boom"):
            self._drive(provider, [ValueError("boom")], monkeypatch)

    def test_all_attempts_exhausted_raises_last_exception(
        self, provider, monkeypatch
    ):
        errors = [_FakeAPIError(503) for _ in range(_RETRY_ATTEMPTS)]
        with pytest.raises(_FakeAPIError) as excinfo:
            self._drive(provider, errors, monkeypatch)
        assert excinfo.value.code == 503

    def test_backoff_capped(self, provider, monkeypatch):
        """The exponential schedule must respect _RETRY_MAX_S so
        attempts late in the loop don't sleep for many minutes."""
        from pilot.providers.gemini_provider import _RETRY_MAX_S

        errors = [_FakeAPIError(503) for _ in range(_RETRY_ATTEMPTS - 1)]
        good = MagicMock(name="response")
        _, _, slept = self._drive(provider, errors + [good], monkeypatch)
        # +25% jitter cap
        for s in slept:
            assert s <= _RETRY_MAX_S * 1.25
