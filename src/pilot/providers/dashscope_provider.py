"""Alibaba DashScope adapter for Qwen3.6-27B (and other open-weights via API).

DashScope's prompt cache is 5-minute ephemeral only — no extended
tier. The audit at thesis-msc/notes/cache_ttl_per_provider.md flags
this as the binding constraint that drives the cross-model
5-minute pinning rule (decision #16).

Token counts come from response.usage:
- input_tokens            → uncached + cached combined
- prompt_tokens_details.cached_tokens → cached_input_tokens (where exposed)
- output_tokens           → output_tokens

DashScope's API is OpenAI-compatible at the `/compatible-mode`
endpoint, but the first-party SDK uses `dashscope.Generation.call(...)`.
This adapter uses the first-party SDK so we get the native usage
fields and can pass DashScope-specific cache config.
"""
from __future__ import annotations

import os
import time

from pilot.providers.base import AnswererProvider, CacheControl, ProviderResult


class DashScopeProvider(AnswererProvider):
    name = "dashscope"

    def __init__(self, api_key: str | None = None) -> None:
        import dashscope

        self._dashscope = dashscope
        api_key_value = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if api_key_value:
            dashscope.api_key = api_key_value

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
        # DashScope context cache (5-min ephemeral) is enabled by default
        # for supported models. CacheControl.DISABLED would require
        # disabling the auto-cache via cache_options, which DashScope's
        # public API doesn't expose; treat DISABLED as best-effort.
        # EPHEMERAL_5MIN matches the native default.
        # EXTENDED_1H / EXTENDED_24H are not available on DashScope;
        # degrade silently to 5min (this is what the cross-model 5-min
        # pin rule is designed to make symmetric across providers).

        params: dict = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        else:
            params["max_tokens"] = 1024

        start = time.perf_counter()
        response = self._dashscope.Generation.call(**params)
        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope call failed: status={response.status_code} "
                f"code={response.code} message={response.message}"
            )

        output = response.output
        usage = response.usage or {}

        # DashScope returns either output.text or output.choices[0].message.content
        # depending on model; handle both.
        text = ""
        if hasattr(output, "text") and output.text:
            text = output.text
        elif hasattr(output, "choices") and output.choices:
            text = output.choices[0].message.content or ""

        # Cached-token reporting on DashScope is in usage.prompt_tokens_details
        # when the model + endpoint expose it; otherwise fall back to 0.
        cached = 0
        details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
        if isinstance(details, dict):
            cached = details.get("cached_tokens", 0) or 0
        total_input = (usage.get("input_tokens", 0) if isinstance(usage, dict) else 0) or 0
        uncached = max(total_input - cached, 0)
        output_tokens = (usage.get("output_tokens", 0) if isinstance(usage, dict) else 0) or 0

        return ProviderResult(
            text=text,
            uncached_input_tokens=uncached,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            provider_request_id=response.request_id,
            wallclock_s=elapsed,
            provider_region=None,
        )
