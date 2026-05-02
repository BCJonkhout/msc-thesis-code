"""Step 1: Model qualification smoke across the candidate slate.

Per pilot plan § 5 Step 1 and § 5.8 row #6:

  Run flat-full-context only (no RAG, no RAPTOR, no GraphRAG), one query
  each, on three smoke contexts (5k, 150k, 600k tokens) against each
  candidate model.

  Decisions out:
    - Eliminate any candidate that fails to return a response within
      10 minutes on the 600k smoke.
    - Eliminate any candidate that emits malformed output on more than
      1 of 3 smokes (no leaked control tags; multiple-choice outputs
      starting with the option letter).

This step does NOT measure quality. It is a binary keep/drop gate
for the cross-model stability check that follows in Phase F.

Behaviour

  - Candidates are loaded from configs/models.yaml. Both the
    `closed_candidates` and `open_weights_candidates` lists are walked.
  - Candidates whose env API key is missing are recorded as `skipped`
    with the missing variable name; not errors.
  - Candidates whose `context_window_tokens` is below a smoke tier's
    requirement are recorded as `context_too_small` for that tier
    only; smaller tiers still run.
  - The cost ledger captures every call. Cost numbers are rolled up
    by candidate at the end.
  - Verdict file: outputs/sanity/step_1_smoke_<utc-timestamp>.json

Exit codes

  0  Every tested candidate passed (or only skipped on context fit).
  1  At least one tested candidate failed the parsing or latency gate.
  2  No candidates had API keys present (nothing was tested).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pilot.ledger import CostLedger, Stage, new_run_id, sha256_hex
from pilot.providers import CacheControl, UnknownProviderError, get_provider
from pilot.providers.base import ProviderResult


# Approximate document sizes. The harness uses 4 chars/token to
# generate a deterministic doc; actual provider tokenisation differs
# by ~25-35% but the size tier is what matters for the pass condition.
_TIER_TOKENS: dict[str, int] = {
    "5k": 5_000,
    "150k": 150_000,
    "600k": 600_000,
}

# Provider env-var lookup. gemini accepts either of two names.
_ENV_VAR: dict[str, str | tuple[str, ...]] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "dashscope": "DASHSCOPE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}

# Pass condition threshold per § 5.8 row #6.
_LATENCY_LIMIT_S = 600  # 10 minutes
_MAX_OUTPUT_TOKENS = 256  # generous; smoke does not need long output

# Patterns that indicate malformed output.
_BAD_TAG_SIGNALS = (
    "<thinking>",
    "</thinking>",
    "<reasoning>",
    "</reasoning>",
    "<scratchpad>",
)


# ─── Smoke document generation ────────────────────────────────────────────────
def _build_smoke_doc(approx_tokens: int) -> str:
    """Deterministic doc of ~approx_tokens tokens (4 chars/token).

    Uses repeating ASCII Lorem-Ipsum so two providers see exactly the
    same prefix bytes; the prompt_hash on the ledger row will be
    stable across runs of the same tier.
    """
    seed = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
        "nisi ut aliquip ex ea commodo consequat. "
    )
    target_chars = approx_tokens * 4
    repeats = max(1, (target_chars // len(seed)) + 1)
    return (seed * repeats)[:target_chars]


_SMOKE_QUERY = (
    "Summarise the document in one sentence. Respond with only a single "
    "concise English sentence and nothing else."
)


def _render_prompt(doc_tokens: int) -> str:
    return f"{_build_smoke_doc(doc_tokens)}\n\nQuestion: {_SMOKE_QUERY}\nAnswer:"


# ─── Verdict shapes ───────────────────────────────────────────────────────────
@dataclass
class TierResult:
    tier: str
    tier_tokens: int
    status: str  # "pass" | "fail_malformed" | "fail_latency" | "context_too_small" | "error"
    wallclock_s: float = 0.0
    uncached_input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    response_first_120: str = ""
    failure_reason: str | None = None


@dataclass
class CandidateVerdict:
    model: str
    provider: str
    context_window_tokens: int | None
    status: str  # "pass" | "fail" | "skipped" | "error"
    reason: str | None = None
    tier_results: list[TierResult] = field(default_factory=list)
    total_usd_estimate: float | None = None


# ─── Resume / idempotency support ─────────────────────────────────────────────
def _load_prior_passes(out_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Walk outputs/sanity/step_1_smoke_*.json and return prior pass entries.

    Returns:
        {model_id: {tier: serialised_tier_result_dict}} for any tier that
        previously achieved status="pass". Most-recent verdict wins on
        conflicts. Tiers that previously failed are NOT included — they
        will be re-run.
    """
    if not out_dir.exists():
        return {}
    prior: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(out_dir.glob("step_1_smoke_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for v in data.get("verdicts", []):
            model_id = v.get("model")
            if not model_id:
                continue
            for tier_dict in v.get("tier_results", []):
                if tier_dict.get("status") == "pass":
                    prior.setdefault(model_id, {})[tier_dict["tier"]] = tier_dict
    return prior


def _tier_result_from_dict(d: dict[str, Any]) -> TierResult:
    """Reconstruct a TierResult from a serialised dict, marking it as reused."""
    return TierResult(
        tier=d["tier"],
        tier_tokens=d.get("tier_tokens", 0),
        status="pass_reused",
        wallclock_s=d.get("wallclock_s", 0.0),
        uncached_input_tokens=d.get("uncached_input_tokens", 0),
        cached_input_tokens=d.get("cached_input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        response_first_120=d.get("response_first_120", ""),
        failure_reason=None,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _env_var_names(provider: str) -> tuple[str, ...]:
    raw = _ENV_VAR.get(provider, "")
    if not raw:
        return ()
    return (raw,) if isinstance(raw, str) else raw


def _provider_has_key(provider: str) -> bool:
    return any(os.environ.get(v) for v in _env_var_names(provider))


def _is_malformed(text: str) -> tuple[bool, str | None]:
    """Return (is_malformed, reason)."""
    if not text or not text.strip():
        return True, "empty_response"
    lc = text.lower()
    for tag in _BAD_TAG_SIGNALS:
        if tag in lc:
            return True, f"leaked_control_tag:{tag}"
    return False, None


def _load_candidates(models_yaml: Path) -> list[dict[str, Any]]:
    """Load all closed + open-weights candidates from models.yaml."""
    with open(models_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    closed = data.get("closed_candidates", []) or []
    opens = data.get("open_weights_candidates", []) or []
    # Filter rejected entries (status=not_accessible etc. handled by skip-key path).
    return [c for c in (closed + opens) if c.get("status") != "rejected"]


def _candidate_context_window(c: dict[str, Any]) -> int | None:
    for key in ("context_window_tokens",
                "context_window_tokens_yarn_extended",
                "context_window_tokens_native"):
        v = c.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return None


# ─── Per-candidate execution ──────────────────────────────────────────────────
def _run_one_tier(
    candidate: dict[str, Any],
    tier: str,
    *,
    ledger: CostLedger,
) -> TierResult:
    """Run one (candidate, tier) call; return a TierResult."""
    tier_tokens = _TIER_TOKENS[tier]
    ctx_window = _candidate_context_window(candidate)
    if ctx_window is not None and tier_tokens > ctx_window:
        return TierResult(
            tier=tier,
            tier_tokens=tier_tokens,
            status="context_too_small",
            failure_reason=f"context_window={ctx_window} < tier_tokens={tier_tokens}",
        )

    provider_name = candidate["provider"]
    model_id = candidate["id"]
    prompt = _render_prompt(tier_tokens)

    try:
        provider = get_provider(provider_name)
    except UnknownProviderError as e:
        return TierResult(
            tier=tier, tier_tokens=tier_tokens, status="error",
            failure_reason=f"unknown_provider:{e}",
        )

    start = time.perf_counter()

    def _call() -> ProviderResult:
        return provider.call(
            prompt,
            model=model_id,
            max_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.0,
            top_p=1.0,
            cache_control=CacheControl.DISABLED,
        )

    try:
        with ledger.log_call(
            architecture="smoke_step_1",
            stage=Stage.GENERATE,
            model=model_id,
            prompt=prompt,
            run_index=0,
            temperature=0.0,
            top_p=1.0,
            max_tokens=_MAX_OUTPUT_TOKENS,
        ) as rec:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                try:
                    result = future.result(timeout=_LATENCY_LIMIT_S)
                except concurrent.futures.TimeoutError:
                    elapsed = time.perf_counter() - start
                    rec.failed = True
                    rec.failure_reason = f"latency_timeout_after_{elapsed:.1f}s"
                    return TierResult(
                        tier=tier, tier_tokens=tier_tokens, status="fail_latency",
                        wallclock_s=elapsed,
                        failure_reason=f"latency_timeout_{_LATENCY_LIMIT_S}s",
                    )

            rec.uncached_input_tokens = result.uncached_input_tokens
            rec.cached_input_tokens = result.cached_input_tokens
            rec.output_tokens = result.output_tokens
            rec.provider_request_id = result.provider_request_id
            rec.response_hash = sha256_hex(result.text)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return TierResult(
            tier=tier, tier_tokens=tier_tokens, status="error",
            wallclock_s=round(elapsed, 4),
            failure_reason=repr(exc),
        )

    elapsed = time.perf_counter() - start
    malformed, reason = _is_malformed(result.text)
    status = "fail_malformed" if malformed else "pass"
    return TierResult(
        tier=tier, tier_tokens=tier_tokens, status=status,
        wallclock_s=round(elapsed, 4),
        uncached_input_tokens=result.uncached_input_tokens,
        cached_input_tokens=result.cached_input_tokens,
        output_tokens=result.output_tokens,
        response_first_120=result.text[:120].replace("\n", " ").strip(),
        failure_reason=reason,
    )


def _aggregate_candidate_status(tier_results: list[TierResult]) -> tuple[str, str | None]:
    """Apply the §5.8 row #6 decision rule to per-tier results.

    `pass_reused` is treated as `pass` (a prior live run already
    cleared the gate for that tier).
    """
    pass_statuses = {"pass", "pass_reused"}

    # Rule 1: 600k latency timeout = eliminate.
    for r in tier_results:
        if r.tier == "600k" and r.status == "fail_latency":
            return "fail", "600k_latency_timeout"
    # Rule 2: malformed on more than 1 of 3 tiers.
    actually_tested = [r for r in tier_results
                       if r.status not in {"context_too_small"}]
    malformed_count = sum(1 for r in actually_tested
                          if r.status == "fail_malformed")
    if malformed_count > 1:
        return "fail", f"malformed_on_{malformed_count}_of_3_smokes"
    # Errors propagate as fail (non-recoverable).
    error_tiers = [r.tier for r in tier_results if r.status == "error"]
    if error_tiers:
        return "fail", f"errors_on_tiers:{','.join(error_tiers)}"
    # Otherwise pass — even if some tiers were context_too_small or reused.
    return "pass", None


# ─── Top-level orchestrator ───────────────────────────────────────────────────
def run_step_1(
    *,
    models_yaml: Path,
    tiers: list[str] | None = None,
    only_candidates: list[str] | None = None,
    out_dir: Path = Path("outputs/sanity"),
    ledger_root: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    candidates = _load_candidates(models_yaml)
    if only_candidates:
        candidates = [c for c in candidates if c["id"] in only_candidates]
    selected_tiers = tiers or list(_TIER_TOKENS.keys())

    run_id = new_run_id()
    ledger = CostLedger(run_id=run_id, root=ledger_root if ledger_root is not None
                        else Path("outputs/runs"))

    prior_passes = _load_prior_passes(out_dir) if resume else {}

    verdicts: list[CandidateVerdict] = []
    for c in candidates:
        provider = c.get("provider", "")
        model_id = c["id"]
        ctx_win = _candidate_context_window(c)

        if not _provider_has_key(provider):
            var_names = " or ".join(_env_var_names(provider)) or "<unknown>"
            verdicts.append(CandidateVerdict(
                model=model_id, provider=provider,
                context_window_tokens=ctx_win,
                status="skipped",
                reason=f"{var_names} not set in environment",
            ))
            continue

        tier_results: list[TierResult] = []
        for tier in selected_tiers:
            prior = prior_passes.get(model_id, {}).get(tier)
            if prior is not None:
                tier_results.append(_tier_result_from_dict(prior))
                continue
            tier_results.append(_run_one_tier(c, tier, ledger=ledger))

        status, reason = _aggregate_candidate_status(tier_results)
        verdicts.append(CandidateVerdict(
            model=model_id, provider=provider,
            context_window_tokens=ctx_win,
            status=status, reason=reason,
            tier_results=tier_results,
        ))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tested = [v for v in verdicts if v.status in {"pass", "fail"}]
    summary = {
        "timestamp_utc": timestamp,
        "run_id": run_id,
        "tiers": selected_tiers,
        "candidates_total": len(verdicts),
        "candidates_tested": len(tested),
        "candidates_skipped": len([v for v in verdicts if v.status == "skipped"]),
        "passed": [v.model for v in verdicts if v.status == "pass"],
        "failed": [v.model for v in verdicts if v.status == "fail"],
        "skipped": [v.model for v in verdicts if v.status == "skipped"],
        "all_passed": bool(tested) and all(v.status == "pass" for v in tested),
        "any_passed": any(v.status == "pass" for v in tested),
        "verdicts": [_serialise_verdict(v) for v in verdicts],
        "ledger_path": str(ledger.path),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_1_smoke_{timestamp}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["verdict_path"] = str(out_path)
    return summary


def _serialise_verdict(v: CandidateVerdict) -> dict[str, Any]:
    d = asdict(v)
    d["tier_results"] = [asdict(t) for t in v.tier_results]
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--models-yaml", type=Path,
        default=Path(__file__).resolve().parents[3] / "configs" / "models.yaml",
    )
    parser.add_argument(
        "--tiers", nargs="+", default=None,
        choices=list(_TIER_TOKENS.keys()),
        help="Subset of context-size tiers to run (default: all three).",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Subset of candidate model IDs (e.g. claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/sanity"),
        help="Directory for the verdict JSON.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Disable resume: re-run every (candidate, tier) even if a prior verdict exists.",
    )
    args = parser.parse_args()

    summary = run_step_1(
        models_yaml=args.models_yaml,
        tiers=args.tiers,
        only_candidates=args.only,
        out_dir=args.out,
        resume=not args.no_resume,
    )

    # Print a tight summary; full verdict is on disk.
    print(json.dumps({k: summary[k] for k in (
        "timestamp_utc", "candidates_total", "candidates_tested",
        "candidates_skipped", "passed", "failed", "skipped",
        "all_passed", "any_passed", "verdict_path",
    )}, indent=2))

    if not summary["candidates_tested"]:
        print(
            "\n[step_1_smoke] No candidates had API keys; nothing tested. "
            "Set the relevant *_API_KEY entries in code/.env and re-run.",
            file=sys.stderr,
        )
        return 2
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
