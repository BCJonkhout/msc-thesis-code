"""Phase F cross-model stability — Kendall's τ between two runs.

Per pilot plan § 7.3.1, the cross-model stability check tests
whether the architecture ranking on the calibration pool is stable
across answerers. Procedure:

  1. Run all 4 architectures × 20 QASPER calibration questions on
     the primary answerer (Gemini 3.1 Pro Preview). This is the
     existing run dir from Step 3.
  2. Repeat on a different answerer (Grok 4.20-0309 / DeepSeek-V4-Pro
     / etc) using ``--answerer-model``. Produces a second run dir.
  3. For each architecture, compute the macro Answer-F1 on QASPER
     under each answerer.
  4. Rank the 4 architectures by macro F1 under each answerer.
  5. Compute Kendall's τ between the two rankings.

Decision rule (pilot plan § 7.3.1):
  τ ≥ 0.67 → architecture rank is stable across answerers
            → primary answerer chosen by cheapest-passing rule
  τ < 0.67 → rank depends on answerer
            → primary answerer chosen by quality, not cost

This module computes the τ and emits a verdict JSON. Pure offline:
takes two run dirs as input, no API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DECISION_THRESHOLD = 0.67


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _macro_f1_per_arch(run_dir: Path) -> dict[str, float]:
    """For each architecture's predictions JSONL in ``run_dir``, return
    macro Answer-F1 over QASPER rows."""
    out: dict[str, float] = {}
    for path in sorted(run_dir.glob("*_predictions.jsonl")):
        arch = path.stem.replace("_predictions", "")
        rows = _load_jsonl(path)
        f1s = [
            r["answer_f1"] for r in rows
            if r.get("dataset") == "qasper" and isinstance(r.get("answer_f1"), (int, float))
        ]
        if f1s:
            out[arch] = sum(f1s) / len(f1s)
    return out


def _rank_by_f1(scores: dict[str, float]) -> list[str]:
    """Return architectures ordered from highest to lowest F1.

    Ties broken by alphabetical name for determinism.
    """
    return [arch for arch, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def _kendall_tau(a: list[str], b: list[str]) -> tuple[float, int, int]:
    """Compute Kendall's τ-b between two complete rankings of the same
    set of items.

    Returns (tau, concordant_pairs, discordant_pairs). τ ranges in
    [-1, 1]; 1 = identical rankings, -1 = exact reverse.
    """
    common = list(set(a) & set(b))
    if len(common) < 2:
        return 0.0, 0, 0
    rank_a = {item: i for i, item in enumerate(a) if item in common}
    rank_b = {item: i for i, item in enumerate(b) if item in common}
    concordant = 0
    discordant = 0
    for i, x in enumerate(common):
        for y in common[i + 1:]:
            sign_a = (rank_a[x] - rank_a[y])
            sign_b = (rank_b[x] - rank_b[y])
            if sign_a * sign_b > 0:
                concordant += 1
            elif sign_a * sign_b < 0:
                discordant += 1
    n_pairs = concordant + discordant
    if n_pairs == 0:
        return 0.0, 0, 0
    tau = (concordant - discordant) / n_pairs
    return tau, concordant, discordant


def compute_phase_f(
    run_a: Path, run_b: Path, *, label_a: str = "A", label_b: str = "B"
) -> dict[str, Any]:
    """Build the Phase F cross-model stability verdict from two run dirs."""
    scores_a = _macro_f1_per_arch(run_a)
    scores_b = _macro_f1_per_arch(run_b)
    common_archs = sorted(set(scores_a) & set(scores_b))

    if len(common_archs) < 2:
        return {
            "error": (
                f"Need ≥ 2 common architectures across both runs to "
                f"compute Kendall's τ. Found: {common_archs}."
            ),
            "scores_a": scores_a,
            "scores_b": scores_b,
        }

    rank_a = _rank_by_f1({k: v for k, v in scores_a.items() if k in common_archs})
    rank_b = _rank_by_f1({k: v for k, v in scores_b.items() if k in common_archs})
    tau, concordant, discordant = _kendall_tau(rank_a, rank_b)

    if tau >= _DECISION_THRESHOLD:
        decision = "STABLE_RANK"
        rationale = (
            f"Kendall's τ = {tau:.3f} ≥ {_DECISION_THRESHOLD:.2f}. "
            "Architecture ranking is stable across the two answerers; "
            "primary answerer can be chosen by the cheapest-passing rule."
        )
    else:
        decision = "RANK_DEPENDS_ON_ANSWERER"
        rationale = (
            f"Kendall's τ = {tau:.3f} < {_DECISION_THRESHOLD:.2f}. "
            "Architecture ranking depends on which answerer is used; "
            "primary answerer should be chosen by quality, not cost."
        )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "run_a": str(run_a),
        "run_b": str(run_b),
        "common_architectures": common_archs,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "rank_a": rank_a,
        "rank_b": rank_b,
        "kendalls_tau": round(tau, 4),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "threshold": _DECISION_THRESHOLD,
        "decision": decision,
        "rationale": rationale,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-a", type=Path, required=True,
                        help="Run dir for the primary answerer")
    parser.add_argument("--run-b", type=Path, required=True,
                        help="Run dir for the alternate answerer")
    parser.add_argument("--label-a", default="primary")
    parser.add_argument("--label-b", default="alternate")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional verdict JSON output path")
    args = parser.parse_args()

    verdict = compute_phase_f(
        args.run_a, args.run_b,
        label_a=args.label_a, label_b=args.label_b,
    )

    if args.out:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        verdict["timestamp_utc"] = timestamp
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(json.dumps(verdict, indent=2))
    return 0 if "error" not in verdict else 1


if __name__ == "__main__":
    sys.exit(main())
