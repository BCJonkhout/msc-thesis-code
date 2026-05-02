"""Generic OpenAI-API-compatible adapter.

Many providers (OpenRouter, Groq, DeepInfra, Together AI, Fireworks AI)
expose an OpenAI-shape `/chat/completions` endpoint at a custom base
URL. Rather than write one adapter per provider, this single class is
parameterised by `base_url` and `api_key_env_var` and reuses the
existing OpenAI Python SDK with a custom client.

Concrete subclasses (e.g. OpenRouterProvider) just set the two
parameters; the call logic is shared.

Caching semantics differ per host:
  - OpenRouter: routes to underlying providers; explicit cache read/write
    pricing exposed when supported. Per-route TTL varies.
  - Groq: ~2h documented cache.
  - DeepInfra / Together / Fireworks: typically no documented cache.

The CacheControl enum maps best-effort: providers without explicit
cache control just see the prompt and rely on whatever automatic
behaviour they have. The cached_input_tokens field on the row is 0
when the host doesn't expose it.
"""
from __future__ import annotations

import os
import time

from pilot.providers.base import AnswererProvider, CacheControl, ProviderResult


class OpenAICompatibleProvider(AnswererProvider):
    """Reusable adapter for any OpenAI-shape /v1/chat/completions endpoint.

    Subclasses set:
        name              short identifier for the factory
        base_url          e.g. "https://openrouter.ai/api/v1"
        api_key_env_var   e.g. "OPENROUTER_API_KEY"
    """

    name: str = "openai-compatible"
    base_url: str = ""
    api_key_env_var: str = ""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        from openai import OpenAI

        if base_url:
            self.base_url = base_url
        if not self.base_url:
            raise ValueError(
                f"{self.__class__.__name__} requires a non-empty base_url; "
                "either set the class attribute or pass base_url=..."
            )

        resolved_key = api_key or (
            os.environ.get(self.api_key_env_var) if self.api_key_env_var else None
        )
        # Lazy-key handling: same pattern as Anthropic/OpenAI/DashScope.
        # Construction succeeds even with no key; the call fails with a
        # provider-side 401 if the key is missing or invalid.
        self._client = OpenAI(api_key=resolved_key, base_url=self.base_url)

    def call(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        cache_control: CacheControl = CacheControl.DISABLED,
    ) -> ProviderResult:
        # OpenAI-compatible endpoints do not universally support
        # prompt_cache_retention. Pass it only if explicitly requested
        # (EXTENDED_24H) and let the host accept-or-ignore. Most hosts
        # will simply ignore unknown kwargs, but some strict gateways
        # reject them; we restrict the kwarg to OpenAI's own host
        # (openai_provider.py).
        extra_kwargs: dict = {}

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens if max_tokens is not None else 1024,
            **extra_kwargs,
        )
        elapsed = time.perf_counter() - start

        usage = response.usage
        cached = 0
        if usage and getattr(usage, "prompt_tokens_details", None) is not None:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        total_input = getattr(usage, "prompt_tokens", 0) or 0
        uncached = max(total_input - cached, 0)
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        text = response.choices[0].message.content or ""

        return ProviderResult(
            text=text,
            uncached_input_tokens=uncached,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            provider_request_id=response.id,
            wallclock_s=elapsed,
            provider_region=None,
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter (https://openrouter.ai) via OpenAI-compatible endpoint.

    OpenRouter routes requests to underlying providers (Groq, DeepInfra,
    Together, etc.) with automatic fallback. Model IDs are slugs like
    `deepseek/deepseek-v4-pro` or `moonshotai/kimi-k2.6`. Caching is
    automatic for OpenAI / DeepSeek / Grok / Groq / Moonshot / Gemini
    2.5 routes (per OpenRouter docs) but NOT for Qwen routes — that's
    why we switched the open-weights row from Qwen to DeepSeek/Moonshot.
    """

    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    api_key_env_var = "OPENROUTER_API_KEY"


class XAIProvider(OpenAICompatibleProvider):
    """xAI (Grok) via OpenAI-compatible endpoint at api.x.ai.

    Model IDs are bare slugs (no prefix): `grok-4.3`,
    `grok-4-1-fast-non-reasoning`, `grok-4.20-0309-non-reasoning`, etc.
    xAI's API closely follows OpenAI's Chat Completions shape; cached
    token counts are returned under prompt_tokens_details.cached_tokens
    when caching fires.
    """

    name = "xai"
    base_url = "https://api.x.ai/v1"
    api_key_env_var = "XAI_API_KEY"
