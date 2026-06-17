"""Provider factory: dispatch by name.

The completed main study resolves ``gemini`` / ``google`` here for the
single answerer (gemini-3.1-flash-lite-preview) and ``xai`` / ``grok``
for the cross-vendor robustness slice. The ``anthropic``, ``openai``,
and ``dashscope`` branches stay wired so the rejected-candidate slate
(configs/models.yaml#rejected_candidates) remains reinstateable without
code changes.
"""
from __future__ import annotations

from pilot.providers.base import AnswererProvider, UnknownProviderError


def get_provider(name: str, **kwargs) -> AnswererProvider:
    """Return a configured AnswererProvider instance by short name.

    Adapters are imported lazily so a missing optional SDK doesn't
    break callers that only need one provider.
    """
    name_lc = name.lower()
    if name_lc == "anthropic":
        from pilot.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(**kwargs)
    if name_lc == "openai":
        from pilot.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(**kwargs)
    if name_lc in {"gemini", "google"}:
        from pilot.providers.gemini_provider import GeminiProvider
        return GeminiProvider(**kwargs)
    if name_lc == "dashscope":
        from pilot.providers.dashscope_provider import DashScopeProvider
        return DashScopeProvider(**kwargs)
    if name_lc == "openrouter":
        from pilot.providers.openai_compatible_provider import OpenRouterProvider
        return OpenRouterProvider(**kwargs)
    if name_lc in {"xai", "grok"}:
        from pilot.providers.openai_compatible_provider import XAIProvider
        return XAIProvider(**kwargs)
    raise UnknownProviderError(
        f"unknown provider {name!r}; expected one of "
        f"anthropic, openai, gemini, dashscope, openrouter, xai"
    )
