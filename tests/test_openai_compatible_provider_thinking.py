"""Tests for the thinking-model headroom workaround in OpenAICompatibleProvider.

Reasoning/thinking models (DeepSeek-V4-Pro, DeepSeek-R1, Grok-4
reasoning variants, Grok-4.3) burn invisible reasoning tokens against
`max_tokens` BEFORE any visible content is emitted. When the cap is
low (default 1024, smoke-phase 256), reasoning consumes the entire
budget and the visible completion truncates to "". Phase G evidence
run measured this at 39/74 (~53%) empty rows for
deepseek/deepseek-v4-pro routed via OpenRouter. The adapter mirrors
the equivalent fix in gemini_provider.py: when the model id matches
a known thinking variant and the requested cap is below 4096, bump
the effective cap to max(max_tokens * 4, 4096) so the visible reply
has headroom.

Coverage:
  - DeepSeek-V4-Pro at max_tokens=256 → cap is bumped to >= 1024
    (specifically max(256*4, 4096) = 4096).
  - GPT-5.4 (not a thinking model) at max_tokens=256 → cap passes
    through unchanged at 256.
  - A thinking model with an explicit large cap (max_tokens=8192) is
    not further inflated — the caller's intent wins.
  - Grok reasoning slugs trigger the bump; the matching non-reasoning
    siblings do NOT (the substring set must discriminate between
    grok-4-fast-reasoning and grok-4-1-fast-non-reasoning, etc.).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
    OpenRouterProvider,
    XAIProvider,
)


def _fake_chat_response(text: str = "ok", request_id: str = "req-1") -> SimpleNamespace:
    """Construct a minimal stand-in for openai.ChatCompletion responses.

    The adapter only reads response.choices[0].message.content,
    response.usage.{prompt_tokens, completion_tokens,
    prompt_tokens_details.cached_tokens}, and response.id, so the
    fake only needs those fields.
    """
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(id=request_id, choices=[choice], usage=usage)


def _provider_with_mock_client(cls=OpenRouterProvider) -> tuple[OpenAICompatibleProvider, MagicMock]:
    """Construct a provider whose internal client.chat.completions.create
    is a MagicMock returning a fake response.

    Passes api_key="test-key" so construction doesn't error when no
    real env var is set in CI.
    """
    provider = cls(api_key="test-key")
    mock_create = MagicMock(return_value=_fake_chat_response())
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create)),
    )
    return provider, mock_create


def test_deepseek_v4_pro_bumps_max_tokens_above_floor() -> None:
    """Thinking model + low cap → effective cap is at least 1024.

    The implementation actually produces max(256*4, 4096) = 4096, but
    the contract for downstream callers is 'enough headroom for the
    visible reply', which we encode loosely as >= 1024.
    """
    provider, mock_create = _provider_with_mock_client(OpenRouterProvider)

    provider.call(
        "hello",
        model="deepseek/deepseek-v4-pro",
        max_tokens=256,
    )

    assert mock_create.call_count == 1
    sent_max_tokens = mock_create.call_args.kwargs["max_tokens"]
    assert sent_max_tokens >= 1024, (
        f"Expected thinking-model headroom bump to lift max_tokens above 1024, "
        f"got {sent_max_tokens}"
    )
    # Tight assertion: the exact formula is max(max_tokens * 4, 4096).
    assert sent_max_tokens == 4096


def test_non_thinking_model_passthrough() -> None:
    """gpt-5.4 is not in the thinking-model substring set → cap unchanged."""
    provider, mock_create = _provider_with_mock_client(OpenRouterProvider)

    provider.call(
        "hello",
        model="gpt-5.4",
        max_tokens=256,
    )

    sent_max_tokens = mock_create.call_args.kwargs["max_tokens"]
    assert sent_max_tokens == 256, (
        f"Expected non-thinking model to pass max_tokens through unchanged, "
        f"got {sent_max_tokens}"
    )


def test_explicit_large_cap_not_further_inflated() -> None:
    """Caller-supplied cap of 8192 on a thinking model is respected as-is.

    The bump only fires when the requested cap is < 4096; above the
    floor the caller has made an informed choice.
    """
    provider, mock_create = _provider_with_mock_client(OpenRouterProvider)

    provider.call(
        "hello",
        model="deepseek/deepseek-v4-pro",
        max_tokens=8192,
    )

    sent_max_tokens = mock_create.call_args.kwargs["max_tokens"]
    assert sent_max_tokens == 8192


@pytest.mark.parametrize(
    "model_id",
    [
        "grok-4-fast-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-4.3",
    ],
)
def test_grok_reasoning_variants_bump(model_id: str) -> None:
    """xAI reasoning slugs trip the thinking-model detector."""
    provider, mock_create = _provider_with_mock_client(XAIProvider)

    provider.call("hello", model=model_id, max_tokens=256)

    sent_max_tokens = mock_create.call_args.kwargs["max_tokens"]
    assert sent_max_tokens >= 1024, (
        f"{model_id}: expected headroom bump, got max_tokens={sent_max_tokens}"
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "grok-4-1-fast-non-reasoning",
        "grok-4.20-0309-non-reasoning",
    ],
)
def test_grok_non_reasoning_variants_passthrough(model_id: str) -> None:
    """The matcher must NOT catch the non-reasoning siblings.

    Both 'grok-4-fast-reasoning' and 'grok-4-1-fast-non-reasoning'
    contain the literal substring 'reasoning'; the implementation
    discriminates with a negative-lookbehind ('(?<!non)-reasoning')
    so only the reasoning variants match. Without that guard the
    non-reasoning slugs would falsely trigger the headroom bump and
    waste cap allocation on models that don't need it.
    """
    provider, mock_create = _provider_with_mock_client(XAIProvider)

    provider.call("hello", model=model_id, max_tokens=256)

    sent_max_tokens = mock_create.call_args.kwargs["max_tokens"]
    assert sent_max_tokens == 256, (
        f"{model_id}: non-reasoning variant must not trip the bump, "
        f"got max_tokens={sent_max_tokens}"
    )
