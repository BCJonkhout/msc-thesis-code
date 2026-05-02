"""OpenAI adapter.

OpenAI prefix-caching is automatic; the `prompt_cache_retention="24h"`
parameter is available on GPT-5.1+ to extend the default short-window
cache to 24 hours. CacheControl.DISABLED cannot truly disable OpenAI's
automatic caching, but the row will simply show low cache hit rate
when no prior identical prefix exists.

Token counts:
- usage.input_tokens             → uncached + cached combined
- usage.input_tokens_details.cached_tokens → cached_input_tokens
- usage.output_tokens            → output_tokens

uncached_input_tokens = input_tokens - cached_tokens.
"""
from __future__ import annotations

import os
import time

from pilot.providers.base import AnswererProvider, CacheControl, ProviderResult


class OpenAIProvider(AnswererProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

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
        extra_kwargs: dict = {}
        if cache_control == CacheControl.EXTENDED_24H:
            extra_kwargs["prompt_cache_retention"] = "24h"
        # OpenAI's automatic caching cannot be turned off; DISABLED is best-effort.

        # GPT-5+ models require `max_completion_tokens` and reject the
        # legacy `max_tokens`. Older OpenAI models accept either. We
        # send max_completion_tokens unconditionally for the OpenAI
        # direct adapter — the OpenAI-compatible adapter (used for
        # OpenRouter, xAI, Groq) keeps `max_tokens` because those
        # gateways follow the older convention.
        max_param = max_tokens if max_tokens is not None else 1024

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_param,
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
