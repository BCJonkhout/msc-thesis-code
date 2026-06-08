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
import re
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
        # Resilience for unattended multi-day runs. The SDK default of 2
        # retries is too shallow for the sustained 429 bursts seen on
        # aggregator routes (xAI / OpenRouter); a 429 that outlasts the
        # retries raises into the sweep dispatcher and leaves the cell with
        # NO prediction row, which the resume path cannot re-drive because
        # it keys off rows that are present. Widen retries and set an
        # explicit timeout sized for long-context generations.
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=self.base_url,
            max_retries=8,
            timeout=120.0,
        )

    # Subclasses can override these to inject provider-specific routing
    # controls. extra_body lands in the JSON body; extra_headers lands as
    # HTTP headers.
    extra_body: dict = {}
    extra_headers: dict = {}

    # Patterns that mark a model id as a reasoning/thinking variant
    # routed through this OpenAI-compatible surface. Thinking models
    # spend invisible reasoning tokens against `max_tokens` BEFORE any
    # visible response is streamed, so a low cap (default 1024 here, or
    # the smoke-phase 256) gets entirely consumed by reasoning and the
    # visible completion truncates to an empty string. Phase G evidence
    # run observed this at 39/74 (~53%) empty rows for
    # deepseek/deepseek-v4-pro routed via OpenRouter. The same headroom
    # bump is applied in gemini_provider.py for the gemini-2.5 /
    # gemini-3 thinking family. Be liberal with detection: false
    # positives just allocate spare cap (cheap, since billing is on
    # output_tokens actually emitted); false negatives silently
    # reproduce the bug.
    #
    # xAI exposes both *-reasoning and *-non-reasoning slugs in the
    # same family, so the matcher must discriminate. Plain substring
    # search for "-reasoning" would falsely catch "*-non-reasoning";
    # the regex below uses a negative-lookbehind so "non-reasoning"
    # tails do not match.
    _THINKING_MODEL_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"deepseek-v4-pro"),
        re.compile(r"deepseek-v4-flash"),
        re.compile(r"deepseek-r1"),
        # Match "-reasoning" only when NOT preceded by "non".
        re.compile(r"(?<!non)-reasoning"),
        # Grok 4.3 ships reasoning-on by default with no separate slug.
        re.compile(r"grok-4\.3"),
    )

    @classmethod
    def _is_thinking_model(cls, model: str) -> bool:
        model_lc = model.lower()
        return any(p.search(model_lc) for p in cls._THINKING_MODEL_PATTERNS)

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
        if self.extra_body:
            extra_kwargs["extra_body"] = dict(self.extra_body)
        if self.extra_headers:
            extra_kwargs["extra_headers"] = dict(self.extra_headers)

        # Thinking-model headroom workaround. See the class-level
        # comment on _THINKING_MODEL_PATTERNS for the failure mode
        # (reasoning tokens consume the cap before any visible content
        # is emitted, truncating the completion to ""). Mirrors the
        # equivalent fix in gemini_provider.py for the gemini-2.5/3
        # family. The caller can still override by passing a
        # max_tokens >= 4096 explicitly.
        effective_max_tokens = max_tokens if max_tokens is not None else 1024
        if self._is_thinking_model(model) and effective_max_tokens < 4096:
            effective_max_tokens = max(effective_max_tokens * 4, 4096)

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=effective_max_tokens,
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
    `deepseek/deepseek-v4-pro` or `moonshotai/kimi-k2.6`.

    Per OpenRouter's prompt-caching best-practices doc, sticky routing
    only activates after a request is observed to use caching. To make
    caching deterministic for the multi-query repeated-context workload
    this pilot measures, the adapter pins a single upstream when a
    cache-supporting upstream exists for the slug. Empirical state
    from `/api/v1/models/<slug>/endpoints` (queried 2026-05-02):

      deepseek/deepseek-v4-pro    : only "deepseek" upstream caches
      deepseek/deepseek-v4-flash  : only "deepseek" upstream caches
      moonshotai/kimi-k2.6        : no upstream supports caching

    Slugs without a caching upstream are routed without a pin (caching
    will not fire there regardless; pinning would only reduce
    availability).
    """

    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    api_key_env_var = "OPENROUTER_API_KEY"

    # Map of model-slug-prefix → upstream tag that supports caching.
    # Slugs not listed here go through with no pin.
    _CACHE_PINNED_UPSTREAMS = {
        "deepseek/": "deepseek",
    }

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
        pinned_upstream = None
        for prefix, upstream in self._CACHE_PINNED_UPSTREAMS.items():
            if model.startswith(prefix):
                pinned_upstream = upstream
                break

        if pinned_upstream:
            self.extra_body = {
                "provider": {
                    "only": [pinned_upstream],
                    "allow_fallbacks": False,
                },
            }
        else:
            self.extra_body = {}

        return super().call(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            cache_control=cache_control,
        )


class XAIProvider(OpenAICompatibleProvider):
    """xAI (Grok) via OpenAI-compatible endpoint at api.x.ai.

    Model IDs are bare slugs (no prefix): `grok-4.3`,
    `grok-4-1-fast-non-reasoning`, `grok-4.20-0309-non-reasoning`, etc.

    xAI's prompt-caching best-practices doc requires the
    `x-grok-conv-id` header for reliable cache hits: cache is
    per-server, requests are load-balanced, and without a stable
    conversation id the second call can land on a different server
    that has never seen the prefix. A stable adapter-instance UUID is
    sent on every call so consecutive calls from the same instance
    target the same server.
    """

    name = "xai"
    base_url = "https://api.x.ai/v1"
    api_key_env_var = "XAI_API_KEY"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        import uuid

        super().__init__(api_key=api_key, base_url=base_url)
        # Per-instance stable id; consecutive calls share it so xAI's
        # load-balancer routes them to the same cache-warm server.
        self.extra_headers = {"x-grok-conv-id": str(uuid.uuid4())}
