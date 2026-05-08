"""Report USD cost for one or more pilot run dirs.

Reads a `CostLedger` JSONL via `pilot.price_card.compute` and emits
a per-architecture, per-stage breakdown plus the total. Useful for
post-hoc cost reporting once a run is closed.

Per pilot plan § 5.8 row #13 (Option A cost-attribution rule), only
``run_index == 0`` rows count toward deployment cost; runs 2..N
exist for variance reporting only and are excluded.

Usage::

    python -m pilot.cli.cost_report --run outputs/runs/<run_id>
    python -m pilot.cli.cost_report --run outputs/runs/A outputs/runs/B
    python -m pilot.cli.cost_report --run outputs/runs/* --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pilot.ledger import CallRecord
from pilot.price_card import _row_cost_usd, compute, load_price_card


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _aggregate_run(run_dir: Path, price_card: dict[str, Any]) -> dict[str, Any]:
    """Per-architecture, per-stage USD breakdown for one run."""
    ledger_path = run_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return {"run_dir": str(run_dir), "error": f"missing {ledger_path}"}

    by_arch_stage: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"rows": 0, "cost_usd": 0.0, "uncached_input_tokens": 0,
                 "cached_input_tokens": 0, "output_tokens": 0}
    )
    failed = 0
    total_rows = 0

    with ledger_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total_rows += 1
            if d.get("run_index", 0) != 0:
                continue
            row = CallRecord(**{k: d.get(k) for k in CallRecord.__dataclass_fields__})
            if row.failed:
                failed += 1
                continue
            cell = by_arch_stage[(row.architecture, row.stage)]
            cell["rows"] += 1
            cell["cost_usd"] += _row_cost_usd(row, price_card)
            cell["uncached_input_tokens"] += row.uncached_input_tokens
            cell["cached_input_tokens"] += row.cached_input_tokens
            cell["output_tokens"] += row.output_tokens

    breakdown = {
        f"{arch}/{stage}": {k: round(v, 6) if isinstance(v, float) else v
                             for k, v in stats.items()}
        for (arch, stage), stats in sorted(by_arch_stage.items())
    }
    total = round(sum(stats["cost_usd"] for stats in by_arch_stage.values()), 6)
    return {
        "run_dir": str(run_dir),
        "ledger_rows_total": total_rows,
        "rows_failed": failed,
        "breakdown": breakdown,
        "total_usd": total,
    }


def _format_table(reports: list[dict[str, Any]]) -> str:
    rows = [
        ["run_id", "arch/stage", "rows", "in_uncached", "in_cached", "out", "usd"]
    ]
    for r in reports:
        if "error" in r:
            rows.append([Path(r["run_dir"]).name, "(error)", "—", "—", "—", "—",
                         r["error"]])
            continue
        run_id = Path(r["run_dir"]).name
        for key, cell in r["breakdown"].items():
            rows.append([
                run_id, key,
                str(cell["rows"]),
                str(cell["uncached_input_tokens"]),
                str(cell["cached_input_tokens"]),
                str(cell["output_tokens"]),
                f"${cell['cost_usd']:.4f}",
            ])
        rows.append([run_id, "TOTAL", "—", "—", "—", "—", f"${r['total_usd']:.4f}"])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(r[i].ljust(widths[i]) for i in range(len(r)))
        for r in rows
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run", nargs="+", type=Path, required=True,
        help="Run directories to report (one or more)",
    )
    parser.add_argument(
        "--price-card", type=Path,
        default=_project_root() / "configs" / "price_card.yaml",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    pc = load_price_card(args.price_card)
    reports = [_aggregate_run(rd, pc) for rd in args.run]

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        print(_format_table(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
