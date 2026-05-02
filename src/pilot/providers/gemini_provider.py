"""Google Gemini adapter via the new google-genai SDK.

google-generativeai (the v0.x package) is deprecated; this adapter
uses the new google-genai package (v1.74+) per
https://github.com/googleapis/python-genai. The client pattern is
`Client(api_key=...)` + `client.models.generate_content(...)`.

Caching is via Gemini's `CachedContent` API. For Step 0 plumbing
this adapter only handles non-cached calls; the cross-model check
will use a separate path that creates a CachedContent with the
appropriate TTL via `client.caches.create(...)` and references it
on the request.

Token counts come from `response.usage_metadata`:
- prompt_token_count          → uncached + cached combined
- cached_content_token_count  → cached_input_tokens
- candidates_token_count      → output_tokens
"""
from __future__ import annotations

import os
import time

from pilot.providers.base import AnswererProvider, CacheControl, ProviderResult


class GeminiProvider(AnswererProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        # google-genai's Client validates the API key at construction time,
        # unlike Anthropic/OpenAI/DashScope which only fail at first call.
        # We defer client construction so factory tests can build the
        # adapter without an API key in the environment; the actual key
        # check happens on the first .call().
        self._api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

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
        from google.genai import types

        client = self._get_client()

        # Step 0 path: no CachedContent. Caching for the cross-model
        # subsample is handled by a higher-level wrapper that creates
        # the CachedContent first and passes its name in via
        # config.cached_content; that path is added when Step 1 wires
        # the cross-model harness.
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens if max_tokens is not None else 1024,
        )

        start = time.perf_counter()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        elapsed = time.perf_counter() - start

        usage = response.usage_metadata
        cached = getattr(usage, "cached_content_token_count", 0) or 0
        total_input = getattr(usage, "prompt_token_count", 0) or 0
        uncached = max(total_input - cached, 0)
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        text = response.text or ""

        return ProviderResult(
            text=text,
            uncached_input_tokens=uncached,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            provider_request_id=getattr(response, "response_id", None),
            wallclock_s=elapsed,
            provider_region=None,
        )
