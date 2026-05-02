"""Provider abstraction base class.

Every concrete adapter (anthropic, openai, gemini, dashscope) returns
a `ProviderResult` with a uniform shape so the cost ledger and price
card see the same fields regardless of provider.

Caching is a first-class concept. Each provider has a different
cache primitive; the abstraction is a `CacheControl` enum that maps
to provider-specific configuration in each adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class UnknownProviderError(KeyError):
    """Raised when get_provider is asked for an unknown provider name."""


class CacheControl(str, Enum):
    """Cache-control intent. Each adapter maps these to its own primitive.

    DISABLED:           No caching at all. Every call pays full uncached cost.
                        Used for the cross-model stability check pinning.
    EPHEMERAL_5MIN:     Default ephemeral cache (5 min on Anthropic,
                        ~minutes on OpenAI automatic, 5 min on DashScope,
                        per-request on Gemini CachedContent).
    EXTENDED_1H:        Anthropic 1h extended cache via beta header.
                        Other providers fall back to their max available.
    EXTENDED_24H:       OpenAI 24h via prompt_cache_retention=24h
                        on GPT-5.1+. Other providers fall back to 1h or 5min.
    """

    DISABLED = "disabled"
    EPHEMERAL_5MIN = "ephemeral_5min"
    EXTENDED_1H = "extended_1h"
    EXTENDED_24H = "extended_24h"


@dataclass(frozen=True)
class ProviderResult:
    """Uniform return shape across providers.

    Token counts come from the provider's response.usage block where
    available; if a provider doesn't expose `cached_input_tokens`,
    the field is 0 (caller can fall back to a heuristic if needed,
    but for verification gates we treat 0 as "no cache hit").
    """

    text: str
    uncached_input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    provider_request_id: str | None
    wallclock_s: float
    provider_region: str | None = None


class AnswererProvider(ABC):
    """Abstract answerer. One subclass per provider."""

    name: str

    @abstractmethod
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
        """Issue one synchronous call and return uniform fields.

        Implementations must:
        - measure wallclock_s themselves (perf_counter delta around the
          API call) so the ledger gets accurate latency.
        - extract token counts from the provider response.
        - return the provider's request id (Anthropic: response.id,
          OpenAI: response.id, Gemini: response.usage_metadata or
          x-goog-request-id, DashScope: response.request_id).
        - map CacheControl to the provider's specific cache primitive.
        """
        raise NotImplementedError
