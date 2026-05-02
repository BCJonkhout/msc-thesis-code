"""Step 2: KV-cache verification per provider.

Per pilot plan § 5 Step 2 and § 5.8 row #1, this orchestrator walks
every provider stack the pilot will use, runs the single-provider
kvcache_check on each, and writes a combined verdict file.

A stack passes if either of these holds (per § 4.3):
  - second-call wallclock <= 0.4 * first-call wallclock, OR
  - cached_input_tokens >= 0.95 * prefix_tokens on the second call.

A stack that fails the gate falls back to the *uncached* cost regime
in the price card; cached cost is set to 0 for that stack and the
flat-full-context baseline reports uncached numbers with a clear note.

Step 2 only verifies the providers whose API keys are present in
the environment. Missing keys produce a 'skipped' status in the
verdict file (not an error) so the pilot can proceed even if not
every key is available simultaneously. Running Step 2 multiple times
as keys are added is fine — each run writes a new timestamped verdict.

Exit code:
  0  All providers with keys passed the gate.
  1  At least one provider with a key failed.
  2  No providers had API keys present (nothing was tested).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pilot.providers import CacheControl, UnknownProviderError, get_provider
from pilot.sanity.kvcache_check import CacheCheckResult, run_cache_check


# Default model per provider for Step 2. The cache primitive is what
# matters here, not which model — but using the cheapest 1M-context
# model per stack keeps the cost bounded.
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6-20260217",
    "openai": "gpt-5.4",
    "gemini": "gemini-3.1-pro-preview",
    "dashscope": "qwen3.6-27b",
}

_ENV_VAR: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
}


def _provider_has_key(name: str) -> bool:
    return bool(os.environ.get(_ENV_VAR[name]))


def _verdict_for_provider(
    name: str,
    *,
    model: str | None,
    doc_tokens: int,
    cache_control: CacheControl,
) -> dict[str, Any]:
    """Run the single-provider check and wrap the result with status fields."""
    if not _provider_has_key(name):
        return {
            "provider": name,
            "status": "skipped",
            "reason": f"{_ENV_VAR[name]} not set in environment",
        }

    use_model = model or _DEFAULT_MODELS[name]
    try:
        result: CacheCheckResult = run_cache_check(
            provider_name=name,
            model=use_model,
            doc_tokens=doc_tokens,
            cache_control=cache_control,
        )
        return {
            "provider": name,
            "status": "pass" if result.overall_pass else "fail",
            "result": asdict(result),
        }
    except UnknownProviderError as e:
        return {"provider": name, "status": "error", "reason": f"unknown provider: {e}"}
    except Exception as e:
        return {"provider": name, "status": "error", "reason": repr(e)}


def run_step_2(
    *,
    providers: list[str] | None = None,
    doc_tokens: int = 100_000,
    cache_control: CacheControl = CacheControl.EPHEMERAL_5MIN,
    out_dir: Path = Path("outputs/sanity"),
) -> dict[str, Any]:
    """Run KV-cache verification across the provider list and write a verdict file.

    The 5-minute ephemeral cache is the default because it is the only
    tier DashScope supports and is the same tier the cross-model check
    pins all providers to. For Step 2 we use the same setting so the
    verdict reflects the regime the cross-model check will run in.

    Returns the verdict dict (also written to disk).
    """
    selected = providers or list(_DEFAULT_MODELS.keys())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    per_provider: list[dict[str, Any]] = []
    for name in selected:
        per_provider.append(
            _verdict_for_provider(
                name,
                model=None,
                doc_tokens=doc_tokens,
                cache_control=cache_control,
            )
        )

    statuses = [p["status"] for p in per_provider]
    tested = [p for p in per_provider if p["status"] in {"pass", "fail", "error"}]
    summary = {
        "timestamp_utc": timestamp,
        "doc_tokens": doc_tokens,
        "cache_control": cache_control.value,
        "providers_requested": selected,
        "providers_tested": [p["provider"] for p in tested],
        "providers_skipped": [p["provider"] for p in per_provider
                              if p["status"] == "skipped"],
        "all_passed": bool(tested) and all(p["status"] == "pass" for p in tested),
        "any_passed": any(p["status"] == "pass" for p in tested),
        "per_provider": per_provider,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_2_kvcache_{timestamp}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["verdict_path"] = str(out_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        choices=list(_DEFAULT_MODELS.keys()),
        help="Subset of providers to verify (default: all).",
    )
    parser.add_argument(
        "--doc-tokens",
        type=int,
        default=100_000,
        help="Approximate document size for the verification (default 100k).",
    )
    parser.add_argument(
        "--cache",
        default="ephemeral_5min",
        choices=[c.value for c in CacheControl],
        help="Cache tier under test (default ephemeral_5min, matching cross-model pin).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sanity"),
        help="Directory to write the verdict JSON.",
    )
    args = parser.parse_args()

    summary = run_step_2(
        providers=args.providers,
        doc_tokens=args.doc_tokens,
        cache_control=CacheControl(args.cache),
        out_dir=args.out,
    )

    print(json.dumps(summary, indent=2))
    print(f"\nWrote verdict: {summary['verdict_path']}")

    if not summary["providers_tested"]:
        print(
            "\n[step_2_kvcache] WARNING: no providers had API keys in env; "
            "nothing was tested. Set at least one of ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY / GOOGLE_API_KEY / DASHSCOPE_API_KEY and re-run.",
            file=sys.stderr,
        )
        return 2
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
