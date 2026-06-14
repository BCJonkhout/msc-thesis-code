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

Retry policy
------------
The Phase G NovelQA sweep runs three answerer lanes concurrently and
pins the RAPTOR/GraphRAG summary stage on Gemini Flash Lite for the
non-Google lanes (multi-provider routing). Under that load Google
returns:

  - 429 / RESOURCE_EXHAUSTED (rate limit) — already retried inside
    the SDK transport, surfaces here only after the SDK gives up.
  - 503 UNAVAILABLE (transient capacity) — *not* retried by the
    SDK, so a single 503 during a GraphRAG entity-extraction sweep
    (~350 calls per novel) bubbles up to the dispatcher as a FAIL,
    wiping the partially-built preprocessing state and forcing a
    full rebuild on the next question.
  - 500 INTERNAL — same shape as 503; transient.

We absorb all three with exponential backoff (1s, 2s, 4s, 8s, 16s)
so the architecture-level FAIL counter only fires on persistent
errors and the preprocessing build can complete despite transient
Google-side throttling.
"""
from __future__ import annotations

import logging
import os
import random
import time

from pilot.providers.base import AnswererProvider, CacheControl, ProviderResult

_log = logging.getLogger(__name__)

# Retryable HTTP status codes. 429: rate limit. 500/502/503/504: server-side
# transient capacity / upstream issues.
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_RETRY_ATTEMPTS = 5
_RETRY_BASE_S = 1.0
_RETRY_MAX_S = 32.0

# Per-request timeout (milliseconds). Without it the underlying httpx client
# has no read timeout, so a single stuck/half-open connection blocks the call
# forever — and under build concurrency that one hung call freezes the whole
# build (the executor waits on it indefinitely). With a timeout, a stuck call
# raises httpx.ReadTimeout/ConnectTimeout, which `_retryable` already treats
# as retryable → exponential-backoff retry instead of an infinite hang.
# Generous enough for the largest flat-over-novel answer call (~hundreds of
# thousands of context tokens) while still bounding a dead connection.
_REQUEST_TIMEOUT_MS = 180_000


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
            from google.genai import types

            # http_options.timeout bounds each request so a stuck connection
            # cannot block a build forever (see _REQUEST_TIMEOUT_MS).
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
            )
        return self._client

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        """True if ``exc`` is a transient API error worth retrying.

        Two classes of failure are absorbed:

        1. ``google.genai.errors.{ServerError,ClientError}`` carrying
           a ``.code`` in {429, 500, 502, 503, 504}. These are the
           overload / capacity signals Google's edge surfaces after
           the SDK's own internal retry has given up.

        2. ``httpx`` transport-level disconnects that bubble through
           the google-genai SDK without an HTTP status: typically
           ``RemoteProtocolError("Server disconnected without
           sending a response.")`` — the server dropped the
           connection mid-request before any response headers came
           back. Observed in production under the NovelQA sweep on
           the FIRST few GraphRAG entity-extraction calls of each
           candidate (idle-pool connection that the Google edge
           closed during the gap between candidates). Also
           ``ReadError``, ``ConnectError``, ``WriteError`` — same
           shape, same correct response (retry with backoff).

        Everything else (JSON parse, schema-validation, auth) stays
        non-retryable; the dispatcher records those as hard FAILs.
        """
        # Class 1: API errors with a retryable status code
        code = getattr(exc, "code", None)
        if isinstance(code, int) and code in _RETRYABLE_CODES:
            return True
        # Class 2: httpx transport-level errors
        try:
            import httpx
        except ImportError:
            return False
        return isinstance(
            exc,
            (
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.ConnectError,
                httpx.WriteError,
                httpx.PoolTimeout,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ),
        )

    def _generate_with_retry(self, *, client, model, prompt, config):
        """Wrap ``client.models.generate_content`` with exponential backoff.

        Sleeps between attempts: 1, 2, 4, 8, 16 s, each with up to ±25%
        jitter so concurrent lanes don't synchronise their retries and
        hit the same 503 window.
        """
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                return client.models.generate_content(
                    model=model, contents=prompt, config=config,
                )
            except Exception as exc:
                if attempt == _RETRY_ATTEMPTS - 1 or not self._retryable(exc):
                    raise
                # For API errors print the status code; for httpx
                # transport errors print the exception class name so
                # the log distinguishes 503-vs-disconnect failure modes.
                signal = getattr(exc, "code", None) or type(exc).__name__
                delay = min(_RETRY_BASE_S * (2 ** attempt), _RETRY_MAX_S)
                delay *= 1.0 + random.uniform(-0.25, 0.25)
                _log.warning(
                    "Gemini call %s on model=%s; sleeping %.1fs before retry "
                    "(attempt %d/%d)",
                    signal, model, delay, attempt + 1, _RETRY_ATTEMPTS,
                )
                time.sleep(delay)
        # Unreachable: the loop either returns or raises.
        raise RuntimeError("retry loop fell through")

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
        #
        # Gemini 3.1 Pro Preview (and Gemini 2.5+ thinking models in
        # general) charge invisible "thinking" tokens against
        # max_output_tokens before any visible response is emitted.
        # With max_output_tokens=1024, JSON-shaped extraction prompts
        # were getting ~990 tokens of thinking and only ~30 tokens of
        # visible response — which truncated the JSON mid-string.
        # Workaround: bump the default cap for thinking models so the
        # visible response has headroom. The caller can still override
        # via the explicit max_tokens kwarg.
        #
        # The ``-latest`` aliases (``gemini-flash-latest``,
        # ``gemini-pro-latest``) currently point at Gemini 2.5 Flash /
        # 2.5 Pro respectively and inherit the same thinking-token
        # accounting, but the explicit ``gemini-2.5`` substring is
        # absent from the alias name itself. Treat the alias forms as
        # thinking models so the QASPER short-answer cap (256) is
        # bumped to the same 4x headroom; without this, every
        # generate-stage call on ``gemini-flash-latest`` returns
        # ~10 visible tokens (the entire 256-token budget is consumed
        # by hidden thinking) and the prediction is silently truncated
        # mid-prose before any answer-bearing text appears.
        model_l = model.lower()
        is_thinking_model = (
            "gemini-3" in model_l
            or "gemini-2.5" in model_l
            or "gemini-flash-latest" in model_l
            or "gemini-pro-latest" in model_l
        )
        effective_max_tokens = max_tokens if max_tokens is not None else 1024
        if is_thinking_model and effective_max_tokens < 4096:
            effective_max_tokens = max(effective_max_tokens * 4, 4096)

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=effective_max_tokens,
        )

        start = time.perf_counter()
        response = self._generate_with_retry(
            client=client, model=model, prompt=prompt, config=config,
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
