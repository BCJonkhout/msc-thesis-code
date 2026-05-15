"""Summarise the NovelQA Phase G grid: cost + completion per candidate.

Iterates over the candidate run_ids identified from the lane logs,
sums the ledger USD by stage, counts predictions per architecture,
and emits a single table the analyst can paste straight into a Pareto
plot. Quality (NovelQA accuracy) comes from a Codabench round-trip;
this script handles only the cost side, which is fully ledger-derived.

Run from ``code/``::

    .venv/Scripts/python.exe scripts/summarise_novelqa_grid.py

Writes ``outputs/sanity/novelqa_grid_summary.json`` and prints a
text table to stdout.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from pilot.ledger import CallRecord
from pilot.price_card import _row_cost_usd, load_price_card


_RID_RE = re.compile(r"run_id=(20\d{6}-[a-f0-9]+)")


def _latest_run_id(log: Path) -> str | None:
    rids: list[str] = []
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = _RID_RE.search(line)
            if m:
                rids.append(m.group(1))
    return rids[-1] if rids else None


def _cost_for_run(run_dir: Path, price_card: dict) -> dict:
    ledger = run_dir / "ledger.jsonl"
    if not ledger.exists():
        return {"error": f"missing {ledger}"}
    by_stage_usd: dict[str, float] = defaultdict(float)
    by_stage_rows: dict[str, int] = defaultdict(int)
    total = 0.0
    failed_rows = 0
    with ledger.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("run_index", 0) != 0:
                continue
            row = CallRecord(**{
                k: d.get(k) for k in CallRecord.__dataclass_fields__
            })
            if row.failed:
                failed_rows += 1
                continue
            c = _row_cost_usd(row, price_card)
            by_stage_usd[row.stage] += c
            by_stage_rows[row.stage] += 1
            total += c
    return {
        "total_usd": round(total, 6),
        "per_stage_usd": {k: round(v, 6) for k, v in by_stage_usd.items()},
        "per_stage_rows": dict(by_stage_rows),
        "failed_rows": failed_rows,
    }


def _arch_counts(run_dir: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in sorted(run_dir.glob("*_predictions.jsonl")):
        arch = path.stem.replace("_predictions", "")
        with path.open(encoding="utf-8") as fh:
            out[arch] = sum(1 for line in fh if line.strip())
    return out


def main() -> int:
    sanity_dir = Path("outputs/sanity")
    runs_dir = Path("outputs/runs")
    logs = sorted(sanity_dir.glob("pareto_novelqa_g_*.log"))
    if not logs:
        print("[summarise] no NovelQA candidate logs found", file=sys.stderr)
        return 1

    price_card = load_price_card(Path("configs/price_card.yaml"))

    rows = []
    for log in logs:
        cand = log.stem.replace("pareto_novelqa_", "")
        rid = _latest_run_id(log)
        if not rid:
            rows.append({"candidate": cand, "error": "no run_id in log"})
            continue
        run_dir = runs_dir / rid
        if not run_dir.exists():
            rows.append({"candidate": cand, "run_id": rid, "error": "run_dir missing"})
            continue
        cost = _cost_for_run(run_dir, price_card)
        arch_counts = _arch_counts(run_dir)
        rows.append({
            "candidate": cand,
            "run_id": rid,
            "arch_counts": arch_counts,
            **cost,
        })

    summary = {
        "schema_version": 1,
        "rows": rows,
    }
    out_path = sanity_dir / "novelqa_grid_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Text table to stdout.
    print(f"{'Candidate':<46} {'flat':>5} {'naive':>6} {'rapt':>5} {'graph':>6} "
          f"{'pre$':>9} {'ret$':>9} {'gen$':>9} {'total$':>9} fail")
    print("-" * 122)
    for r in rows:
        if "error" in r:
            print(f"{r['candidate']:<46}  ERROR: {r['error']}")
            continue
        ac = r["arch_counts"]
        ps = r["per_stage_usd"]
        print(
            f"{r['candidate']:<46} "
            f"{ac.get('flat', 0):>5} "
            f"{ac.get('naive_rag', 0):>6} "
            f"{ac.get('raptor', 0):>5} "
            f"{ac.get('graphrag', 0):>6} "
            f"{ps.get('preprocess', 0.0):>9.4f} "
            f"{ps.get('retrieval', 0.0):>9.4f} "
            f"{ps.get('generate', 0.0):>9.4f} "
            f"{r['total_usd']:>9.4f} "
            f"{r['failed_rows']:>4}"
        )
    print()
    print(f"[summarise] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
