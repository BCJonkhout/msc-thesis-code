"""KV-cache reuse verification harness.

Per pilot plan § 4.3 and § 5.8 row #1, this script verifies that
prompt caching actually fires on a given provider stack.

Procedure:
    1. Send (toy_document + Q1) to the provider with caching enabled.
       Record prefill latency + cached_input_tokens.
    2. Immediately send (toy_document + Q2) with caching enabled.
       Record prefill latency + cached_input_tokens.

Pass condition (either is sufficient):
    A. Second-call wallclock <= 0.4 * first-call wallclock, OR
    B. Provider-returned cached_input_tokens on the second call
       >= 0.95 * len(toy_document_tokens).

Failure means the cache headers are wrong or the cache wasn't
exercised correctly. The next pipeline step (real flat-full-context
runs) cannot be trusted until this passes.

This is the Step 0 plumbing-only check; full per-stack sweep with
all four closed providers + DashScope happens in Step 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from pilot.providers import CacheControl, get_provider


@dataclass
class CacheCheckResult:
    provider: str
    model: str
    cache_control: str
    doc_tokens_estimate: int
    first_call_wallclock_s: float
    first_call_uncached_tokens: int
    first_call_cached_tokens: int
    second_call_wallclock_s: float
    second_call_uncached_tokens: int
    second_call_cached_tokens: int
    latency_ratio: float
    latency_pass: bool
    cached_token_ratio: float
    cached_token_pass: bool
    overall_pass: bool


def _build_toy_document(approx_tokens: int) -> str:
    """Build a deterministic toy document of ~approx_tokens tokens.

    Uses repeating ASCII Lorem-Ipsum so two providers see exactly the
    same prefix. Token counting is rough (1 token ≈ 4 chars) but
    consistent for our threshold logic.
    """
    seed_paragraph = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
        "nisi ut aliquip ex ea commodo consequat. "
    )
    target_chars = approx_tokens * 4
    repeats = max(1, (target_chars // len(seed_paragraph)) + 1)
    return (seed_paragraph * repeats)[:target_chars]


def run_cache_check(
    *,
    provider_name: str,
    model: str,
    doc_tokens: int = 100_000,
    cache_control: CacheControl = CacheControl.EPHEMERAL_5MIN,
) -> CacheCheckResult:
    """Issue Q1 and Q2 over the same toy doc; return the verdict."""
    provider = get_provider(provider_name)
    doc = _build_toy_document(doc_tokens)

    q1 = "Summarise the document in one sentence."
    q2 = "List the main themes of the document."

    prompt1 = f"{doc}\n\nQ1: {q1}\nA:"
    prompt2 = f"{doc}\n\nQ2: {q2}\nA:"

    r1 = provider.call(
        prompt1,
        model=model,
        max_tokens=128,
        cache_control=cache_control,
    )
    r2 = provider.call(
        prompt2,
        model=model,
        max_tokens=128,
        cache_control=cache_control,
    )

    latency_ratio = r2.wallclock_s / r1.wallclock_s if r1.wallclock_s > 0 else float("inf")
    cached_token_ratio = (
        r2.cached_input_tokens / max(doc_tokens, 1)
    )

    latency_pass = latency_ratio <= 0.4
    cached_token_pass = cached_token_ratio >= 0.95
    overall_pass = latency_pass or cached_token_pass

    return CacheCheckResult(
        provider=provider_name,
        model=model,
        cache_control=cache_control.value,
        doc_tokens_estimate=doc_tokens,
        first_call_wallclock_s=round(r1.wallclock_s, 4),
        first_call_uncached_tokens=r1.uncached_input_tokens,
        first_call_cached_tokens=r1.cached_input_tokens,
        second_call_wallclock_s=round(r2.wallclock_s, 4),
        second_call_uncached_tokens=r2.uncached_input_tokens,
        second_call_cached_tokens=r2.cached_input_tokens,
        latency_ratio=round(latency_ratio, 4),
        latency_pass=latency_pass,
        cached_token_ratio=round(cached_token_ratio, 4),
        cached_token_pass=cached_token_pass,
        overall_pass=overall_pass,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", required=True,
                        choices=["anthropic", "openai", "gemini", "dashscope"])
    parser.add_argument("--model", required=True,
                        help="Exact dated model snapshot (e.g. claude-sonnet-4-6-20260217)")
    parser.add_argument("--doc-tokens", type=int, default=100_000,
                        help="Approximate document size in tokens (default 100,000)")
    parser.add_argument("--cache", default="ephemeral_5min",
                        choices=[c.value for c in CacheControl])
    parser.add_argument("--out", default="outputs/sanity",
                        help="Output directory for the verdict JSON")
    args = parser.parse_args()

    cache = CacheControl(args.cache)
    result = run_cache_check(
        provider_name=args.provider,
        model=args.model,
        doc_tokens=args.doc_tokens,
        cache_control=cache,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"kvcache_{args.provider}_{timestamp}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    print(json.dumps(asdict(result), indent=2))
    print(f"\nWrote verdict: {out_path}")
    return 0 if result.overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
