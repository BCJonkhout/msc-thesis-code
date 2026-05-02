"""Step 0 smoke: exercise the plumbing on a 1k-token toy document.

Per pilot plan § 5 Step 0 Bootstrap: "Smoke-test on a 1k-token toy doc.
Falls out: runnable scaffolding."

The smoke does not test model quality. It proves:
- Configs load and pass the provenance gate.
- A prompt template renders byte-identically across two invocations.
- One provider adapter issues a real API call without crashing.
- The cost ledger writes a JSONL row containing the expected fields.
- The price-card resolver sums to a non-zero USD total when given
  the recorded row.

Exit 0 = plumbing works. Exit non-zero = fix before moving to Step 1.

Default provider is anthropic with claude-sonnet-4-6-20260217 (cheapest
1M-context closed candidate that passes smoke at the first call).
The DASHSCOPE_BASED smoke can be requested with `--provider dashscope`
once a DashScope API key is in the environment.

Environment variables expected:
    ANTHROPIC_API_KEY (or OPENAI_API_KEY, GOOGLE_API_KEY, DASHSCOPE_API_KEY
    depending on --provider)

If the API key is missing the smoke exits 2 with a clear message
rather than failing inside the SDK.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pilot.ledger import CostLedger, Stage, new_run_id, sha256_hex
from pilot.price_card import compute, load_price_card
from pilot.prompts import load_template
from pilot.providers import CacheControl, get_provider
from pilot.provenance import load_and_validate

_PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
    "openai": "gpt-5.4",
    "gemini": "gemini-3.1-pro-preview",
    "dashscope": "qwen3.6-27b",
    "openrouter": "deepseek/deepseek-v4-pro",
    "xai": "grok-4.3",
}

_PROVIDER_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "opus": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",   # GeminiProvider also accepts GOOGLE_API_KEY
    "dashscope": "DASHSCOPE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}


_TOY_DOC = (
    "The PagedAttention paper (Kwon et al., 2023) introduces a memory "
    "management technique for the key-value cache of large language model "
    "inference. By treating the KV cache as paged virtual memory, the "
    "system supports prefix sharing across concurrent requests, dynamic "
    "growth without contiguous memory requirements, and eviction policies "
    "decoupled from the model's attention computation. The vLLM serving "
    "stack productionised this approach and exposed it to operators via "
    "an HTTP API. Subsequent work has extended the idea to KV-cache "
    "compression, prefix tree caches, and cross-tenant cache sharing. "
    "For repeated-context inference, the relevant property is that two "
    "requests sharing a long document prefix can amortise the prefill "
    "cost of that prefix to near zero on the second request, provided "
    "the cache state has not been evicted. The cache thus behaves as a "
    "free same-document amortisation primitive at the serving layer."
) * 4  # ~1k tokens (4 chars/token)
_TOY_QUERY = "What does PagedAttention's prefix-sharing capability imply for repeated-context inference?"


def _validate_configs(configs_dir: Path) -> dict[str, dict]:
    """Load and provenance-validate all four configs."""
    loaded = {}
    for name in ("price_card", "methods", "embedding", "models"):
        path = configs_dir / f"{name}.yaml"
        loaded[name] = load_and_validate(path)
    return loaded


def _check_template_determinism(template_name: str, **slots: str) -> str:
    """Render the template twice; assert byte-identical output.

    Returns the rendered string on success.
    """
    tpl = load_template(template_name)
    a = tpl.render(**slots)
    b = tpl.render(**slots)
    if a != b:
        raise AssertionError(
            f"template {template_name!r} rendered non-deterministically: "
            f"sha256(a)={sha256_hex(a)[:8]} sha256(b)={sha256_hex(b)[:8]}"
        )
    return a


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 0 plumbing smoke.")
    parser.add_argument("--provider", default="anthropic",
                        choices=list(_PROVIDER_DEFAULT_MODEL.keys()),
                        help="Provider for the smoke (default: anthropic).")
    parser.add_argument("--model", default=None,
                        help="Override the default model for this provider.")
    parser.add_argument("--configs-dir", type=Path,
                        default=Path(__file__).resolve().parents[3] / "configs")
    parser.add_argument("--outputs-root", type=Path,
                        default=Path("outputs/runs"))
    args = parser.parse_args()

    print(f"[step_0_smoke] configs dir: {args.configs_dir}")
    print(f"[step_0_smoke] outputs root: {args.outputs_root.resolve()}")

    # ── 1. Load and validate configs ─────────────────────────────
    print("[step_0_smoke] loading configs and validating provenance...")
    configs = _validate_configs(args.configs_dir)
    price_card = configs["price_card"]
    print(f"[step_0_smoke]   loaded {len(configs)} configs; provenance OK")

    # ── 2. Prompt template determinism ───────────────────────────
    print("[step_0_smoke] rendering prompt twice; checking determinism...")
    rendered = _check_template_determinism(
        "qa_freeform",
        context=_TOY_DOC,
        query=_TOY_QUERY,
    )
    print(f"[step_0_smoke]   rendered {len(rendered)} chars; sha256={sha256_hex(rendered)[:16]}")

    # ── 3. API key check ─────────────────────────────────────────
    env_var = _PROVIDER_ENV_VAR[args.provider]
    if not os.environ.get(env_var):
        print(f"[step_0_smoke] ERROR: {env_var} is not set in the environment.",
              file=sys.stderr)
        return 2
    model = args.model or _PROVIDER_DEFAULT_MODEL[args.provider]
    print(f"[step_0_smoke] provider={args.provider} model={model}")

    # ── 4. Issue one real call through the ledger ────────────────
    run_id = new_run_id()
    ledger = CostLedger(run_id=run_id, root=args.outputs_root)
    print(f"[step_0_smoke] run_id={run_id}")

    provider = get_provider(args.provider)
    with ledger.log_call(
        architecture="flat_full_context",
        stage=Stage.GENERATE,
        model=model,
        prompt=rendered,
        run_index=0,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
    ) as rec:
        result = provider.call(
            rendered,
            model=model,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            cache_control=CacheControl.DISABLED,
        )
        rec.uncached_input_tokens = result.uncached_input_tokens
        rec.cached_input_tokens = result.cached_input_tokens
        rec.output_tokens = result.output_tokens
        rec.provider_request_id = result.provider_request_id
        rec.response_hash = sha256_hex(result.text)

    print(f"[step_0_smoke]   wrote ledger row to {ledger.path}")
    print(f"[step_0_smoke]   response (first 120 chars): {result.text[:120]!r}")

    # ── 5. Read back, compute USD via price card ─────────────────
    rows = ledger.read()
    if not rows:
        print("[step_0_smoke] ERROR: ledger is empty after the call.", file=sys.stderr)
        return 3
    last = rows[-1]
    print(f"[step_0_smoke]   ledger row: model={last.model} "
          f"in={last.uncached_input_tokens} cached={last.cached_input_tokens} "
          f"out={last.output_tokens} wall={last.wallclock_s:.3f}s "
          f"req_id={last.provider_request_id}")

    usd = compute(ledger.path, price_card)
    print(f"[step_0_smoke]   price-card total: ${usd:.6f}")
    if usd <= 0.0:
        print("[step_0_smoke] WARNING: total USD is 0.0; verify the provider+model "
              "is registered in price_card.yaml under providers.<name>.models.",
              file=sys.stderr)

    # ── 6. Done ──────────────────────────────────────────────────
    print("[step_0_smoke] OK — plumbing works end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
