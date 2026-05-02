"""Anthropic adapter.

Maps `CacheControl` to Anthropic's `cache_control: ephemeral` block
on the last message content. The 1h extended tier requires the
`extended-cache-ttl-2025-04-11` beta header plus `cache_control.ttl="1h"`
per the Anthropic prompt-caching docs.

Token counts come from response.usage:
- input_tokens          → uncached_input_tokens
- cache_read_input_tokens → cached_input_tokens
- cache_creation_input_tokens (one-time write cost; not counted as
  cached_input on the row that creates the cache)
- output_tokens         → output_tokens
"""
from __future__ import annotations

import os
import time

from pilot.providers.base import AnswererProvider, CacheControl, ProviderResult


class AnthropicProvider(AnswererProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
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
        content_block: dict = {"type": "text", "text": prompt}

        extra_headers: dict[str, str] = {}
        if cache_control == CacheControl.EPHEMERAL_5MIN:
            content_block["cache_control"] = {"type": "ephemeral"}
        elif cache_control == CacheControl.EXTENDED_1H:
            content_block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
            extra_headers["anthropic-beta"] = "extended-cache-ttl-2025-04-11"
        elif cache_control == CacheControl.EXTENDED_24H:
            # Anthropic max is 1h; degrade to 1h.
            content_block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
            extra_headers["anthropic-beta"] = "extended-cache-ttl-2025-04-11"
        # DISABLED: leave the block bare; no cache hit will register.

        # Anthropic rejects sending both temperature and top_p on newer
        # Claude models (Sonnet 4.6+, Opus 4.7+). With T=0 (greedy) top_p
        # is irrelevant, so we send only temperature in that case. For
        # T>0 we send only top_p when it differs from the default 1.0.
        sampling_kwargs: dict = {}
        if temperature == 0.0:
            sampling_kwargs["temperature"] = 0.0
        elif top_p != 1.0:
            sampling_kwargs["top_p"] = top_p
        else:
            sampling_kwargs["temperature"] = temperature

        start = time.perf_counter()
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens if max_tokens is not None else 1024,
            messages=[{"role": "user", "content": [content_block]}],
            extra_headers=extra_headers or None,
            **sampling_kwargs,
        )
        elapsed = time.perf_counter() - start

        usage = response.usage
        text_parts = [block.text for block in response.content if block.type == "text"]
        text = "".join(text_parts)

        return ProviderResult(
            text=text,
            uncached_input_tokens=getattr(usage, "input_tokens", 0) or 0,
            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            provider_request_id=response.id,
            wallclock_s=elapsed,
            provider_region=None,
        )
