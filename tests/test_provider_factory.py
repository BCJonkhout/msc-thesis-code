"""Provider factory: each adapter is reachable; uniform interface; no API calls."""
from __future__ import annotations

import pytest

from pilot.providers import (
    AnswererProvider,
    UnknownProviderError,
    get_provider,
)


@pytest.mark.parametrize("name", ["anthropic", "openai", "gemini", "dashscope", "openrouter"])
def test_factory_returns_provider(name: str) -> None:
    """Each adapter resolves to an AnswererProvider subclass without making any API call.

    We only construct the client; we don't call .call(). Construction
    can fail if the SDK is missing — that surfaces as ImportError,
    which is a real environment problem worth seeing in the test
    output rather than silently passing.
    """
    provider = get_provider(name)
    assert isinstance(provider, AnswererProvider)
    assert provider.name == ("gemini" if name in {"gemini", "google"} else name)


def test_unknown_provider_raises() -> None:
    """An unknown provider name raises UnknownProviderError."""
    with pytest.raises(UnknownProviderError):
        get_provider("not-a-real-provider")


def test_provider_call_signature() -> None:
    """Every adapter's call method exposes the same keyword-argument shape."""
    import inspect

    from pilot.providers.base import AnswererProvider

    expected_kwargs = {
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "cache_control",
    }
    for name in ("anthropic", "openai", "gemini", "dashscope", "openrouter"):
        provider = get_provider(name)
        sig = inspect.signature(provider.call)
        kwargs_in_sig = {
            n for n, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.KEYWORD_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and n != "self" and n != "prompt"
        }
        missing = expected_kwargs - kwargs_in_sig
        assert not missing, f"{name}.call missing kwargs: {missing}"
