"""Phase F Pareto aggregator — cost-vs-quality across all candidate answerers.

Reads multiple Step-3-dry-run verdicts (one per candidate × architecture
slate) plus a single Gemini Pro reference run dir, and emits a unified
cost-vs-quality table:

  - per-architecture macro Answer-F1 for each candidate
  - Kendall's τ-b vs the Gemini Pro reference's architecture ranking
  - cheapest-passing decision (τ ≥ 0.67 → STABLE_RANK; else
    RANK_DEPENDS_ON_ANSWERER)
  - total USD cost from the candidate's ledger (run_index = 0 only,
    per the Option A cost-attribution rule)

Usage::

    python -m pilot.cli.phase_f_pareto \\
        --reference outputs/runs/_consolidated_gemini_pro_4arch \\
        --reference-label gemini-3.1-pro-preview \\
        --candidates outputs/runs/<run_a> outputs/runs/<run_b> ... \\
        --out outputs/sanity/phase_f_pareto.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pilot.cli.phase_f_kendall import _kendall_tau, _macro_f1_per_arch, _rank_by_f1
from pilot.ledger import CallRecord
from pilot.price_card import _row_cost_usd, load_price_card


_DECISION_THRESHOLD = 0.67


def _aggregate_cost(run_dir: Path, price_card: dict[str, Any]) -> dict[str, float]:
    """Total USD + per-stage breakdown for one run dir, run_index=0 only."""
    ledger_path = run_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return {"error": f"missing {ledger_path}"}

    by_stage: dict[str, float] = defaultdict(float)
    total = 0.0
    rows = 0
    with ledger_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("run_index", 0) != 0:
                continue
            row = CallRecord(**{k: d.get(k) for k in CallRecord.__dataclass_fields__})
            if row.failed:
                continue
            cost = _row_cost_usd(row, price_card)
            by_stage[row.stage] += cost
            total += cost
            rows += 1

    return {
        "total_usd": round(total, 6),
        "per_stage_usd": {k: round(v, 6) for k, v in by_stage.items()},
        "ledger_rows": rows,
    }


def _candidate_label_from_run_dir(run_dir: Path) -> str:
    """Read the most-recent verdict pointing at this run_dir to get the
    answerer_model. Falls back to the directory name when no verdict
    points at it."""
    sanity_dir = run_dir.parent.parent / "sanity"
    if not sanity_dir.exists():
        return run_dir.name
    candidates = sorted(sanity_dir.glob("step_3_dry_run_*.json"), reverse=True)
    for vp in candidates:
        try:
            v = json.loads(vp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(v.get("predictions_dir", "")).name == run_dir.name:
            return v.get("answerer_model") or run_dir.name
    return run_dir.name


def build_pareto_table(
    candidates: list[Path],
    reference: Path,
    *,
    reference_label: str = "reference",
    price_card: dict[str, Any],
) -> dict[str, Any]:
    """Combine per-candidate F1 / cost / Kendall verdicts into one table."""
    ref_scores = _macro_f1_per_arch(reference)
    ref_rank = _rank_by_f1(ref_scores)

    rows: list[dict[str, Any]] = []
    for cd in candidates:
        scores = _macro_f1_per_arch(cd)
        if not scores:
            rows.append({"run_dir": str(cd), "error": "no predictions found"})
            continue
        common_archs = sorted(set(scores) & set(ref_scores))
        if len(common_archs) < 2:
            rows.append({
                "run_dir": str(cd),
                "label": _candidate_label_from_run_dir(cd),
                "scores": scores,
                "error": (
                    f"need ≥ 2 common architectures vs reference; "
                    f"found {common_archs}"
                ),
            })
            continue
        cd_rank = _rank_by_f1({k: v for k, v in scores.items() if k in common_archs})
        ref_rank_common = [a for a in ref_rank if a in common_archs]
        tau, conc, disc = _kendall_tau(ref_rank_common, cd_rank)

        cost = _aggregate_cost(cd, price_card)

        decision = "STABLE_RANK" if tau >= _DECISION_THRESHOLD else "RANK_DEPENDS_ON_ANSWERER"

        rows.append({
            "run_dir": str(cd),
            "label": _candidate_label_from_run_dir(cd),
            "common_architectures": common_archs,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "rank": cd_rank,
            "kendalls_tau_vs_reference": round(tau, 4),
            "concordant_pairs": conc,
            "discordant_pairs": disc,
            "decision": decision,
            "total_usd": cost.get("total_usd"),
            "per_stage_usd": cost.get("per_stage_usd"),
            "ledger_rows": cost.get("ledger_rows"),
        })

    # Sort by cost ascending so the cheapest-passing candidate sits at the top
    rows.sort(key=lambda r: (
        r.get("total_usd", float("inf")) if isinstance(r.get("total_usd"), (int, float))
        else float("inf")
    ))

    return {
        "reference": {
            "run_dir": str(reference),
            "label": reference_label,
            "scores": {k: round(v, 4) for k, v in ref_scores.items()},
            "rank": ref_rank,
        },
        "candidates": rows,
        "threshold": _DECISION_THRESHOLD,
    }


def _format_table(verdict: dict[str, Any]) -> str:
    """Render the Pareto verdict as a fixed-width plain-text table."""
    archs = ["flat", "naive_rag", "raptor", "graphrag"]
    header = ["candidate", *archs, "τ", "USD", "decision"]
    rows = [header]
    ref = verdict["reference"]
    rows.append([
        f"[ref] {ref['label']}",
        *(f"{ref['scores'].get(a, '—'):.3f}" if isinstance(ref['scores'].get(a), (int, float))
          else "—" for a in archs),
        "—", "—", "—",
    ])
    for r in verdict["candidates"]:
        if "error" in r:
            rows.append([r.get("label", Path(r["run_dir"]).name),
                        "—", "—", "—", "—", "—", "—", r["error"][:32]])
            continue
        rows.append([
            r["label"],
            *(f"{r['scores'].get(a):.3f}" if isinstance(r['scores'].get(a), (int, float))
              else "—" for a in archs),
            f"{r['kendalls_tau_vs_reference']:+.2f}",
            f"${r['total_usd']:.4f}" if r.get('total_usd') is not None else "—",
            r["decision"][:24],
        ])
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(r))) for r in rows
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, required=True,
                        help="Run dir for the reference answerer "
                             "(typically the consolidated Gemini Pro 4-arch run)")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidates", nargs="+", type=Path, required=True,
                        help="Run dirs to compare against the reference")
    parser.add_argument("--price-card", type=Path,
                        default=_project_root() / "configs" / "price_card.yaml")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of the text table")
    args = parser.parse_args()

    pc = load_price_card(args.price_card)
    verdict = build_pareto_table(
        args.candidates, args.reference,
        reference_label=args.reference_label, price_card=pc,
    )

    if args.json:
        print(json.dumps(verdict, indent=2))
    else:
        print(_format_table(verdict))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
