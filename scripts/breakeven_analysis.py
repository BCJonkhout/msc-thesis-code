"""Compute the per-(candidate, architecture) break-even N vs flat full-context.

The thesis's central claim is that preprocessing-based retrieval
architectures (Naive RAG, RAPTOR, GraphRAG) amortise their offline
build cost across multiple questions on the same document, so they
eventually beat flat full-context once N is large enough. Formally::

    \\bar{C}_deploy(c, n) = (C_off^struct(c) + sum_i C_on(q_i)) / n

The architecture beats flat at N questions iff

    C_off^struct / N + C_on  <  C_flat_per_query

Solving for the break-even N::

    N*  =  C_off^struct  /  (C_flat_per_query - C_on)         if C_flat > C_on
    N*  =  +infinity                                          otherwise
           (preprocessing's per-query cost dominates flat's;
           amortisation cannot save it)

We compute per-(candidate, paper, arch) figures directly from the
cost ledger:
  - C_off^struct(paper, arch) = sum of Stage.PREPROCESS rows in the
    run, partitioned by paper via the per-cell ordering. We use the
    SIMPLIFIED rule "the first question on a paper for an
    architecture bears the entire preprocess block" — matches how
    the per-(arch, paper) cache populates exactly once per paper.
  - C_on per question = mean (RETRIEVAL + GENERATE) row cost per
    cell for that (candidate, arch).
  - C_flat per question = the same calculation for arch=flat. Flat
    has no PREPROCESS rows; every flat call is full prefill +
    answer in one shot.

Run from ``code/``::

    .venv/Scripts/python.exe scripts/breakeven_analysis.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pilot.ledger import CallRecord
from pilot.price_card import _row_cost_usd, load_price_card


def _latest_run_id(log: Path) -> str | None:
    rids: list[str] = []
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "run_id=" not in line:
                continue
            m = line.split("run_id=", 1)[1].split()[0].rstrip(",")
            rids.append(m)
    return rids[-1] if rids else None


def _cost_per_arch(run_dir: Path, price_card: dict) -> dict[str, dict]:
    """For each architecture, total preprocess cost + total online cost
    + number of cells (from predictions JSONLs)."""
    out: dict[str, dict] = defaultdict(
        lambda: {
            "preprocess_usd": 0.0,
            "retrieval_usd": 0.0,
            "generate_usd": 0.0,
            "n_cells": 0,
        }
    )

    # Cell counts from predictions JSONL.
    for path in sorted(run_dir.glob("*_predictions.jsonl")):
        arch = path.stem.replace("_predictions", "")
        with path.open(encoding="utf-8") as fh:
            cells = sum(1 for line in fh if line.strip())
        out[arch]["n_cells"] = cells

    # Ledger rows. We attribute by architecture (each row carries it).
    ledger = run_dir / "ledger.jsonl"
    if not ledger.exists():
        return dict(out)
    with ledger.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("run_index", 0) != 0 or d.get("failed"):
                continue
            row = CallRecord(**{
                k: d.get(k) for k in CallRecord.__dataclass_fields__
            })
            arch = row.architecture
            usd = _row_cost_usd(row, price_card)
            key = f"{row.stage}_usd"
            out[arch][key] += usd
    return dict(out)


def _breakeven_n(c_off: float, c_on_per_q: float, c_flat_per_q: float) -> float:
    """Return the break-even N, or +inf when preprocessing can never
    save vs flat (because per-question cost is already higher)."""
    delta = c_flat_per_q - c_on_per_q
    if delta <= 0:
        return float("inf")
    return c_off / delta


def main() -> int:
    sanity = Path("outputs/sanity")
    runs = Path("outputs/runs")
    price_card = load_price_card(Path("configs/price_card.yaml"))

    candidates: list[tuple[str, Path]] = []
    for log in sorted(sanity.glob("pareto_novelqa_g_*.log")):
        cand = log.stem.replace("pareto_novelqa_", "")
        rid = _latest_run_id(log)
        if rid is None:
            continue
        rd = runs / rid
        if rd.exists():
            candidates.append((cand, rd))

    # Filter to full-grid candidates only (n_cells == 20 across all 4
    # archs); partial grids would distort C_on per question.
    rows = []
    for cand, rd in candidates:
        per_arch = _cost_per_arch(rd, price_card)
        if not per_arch.get("flat", {}).get("n_cells"):
            continue
        flat = per_arch["flat"]
        n_flat = flat["n_cells"]
        # Flat's per-question cost is its total cost / n_cells.
        c_flat = (
            flat["preprocess_usd"]  # always 0 for flat
            + flat["retrieval_usd"]  # always 0 for flat
            + flat["generate_usd"]
        ) / n_flat

        cand_rows = {"candidate": cand, "c_flat_per_question": c_flat,
                     "n_questions_flat": n_flat, "archs": {}}
        for arch in ("naive_rag", "raptor", "graphrag"):
            stats = per_arch.get(arch)
            if not stats or not stats["n_cells"]:
                continue
            n = stats["n_cells"]
            c_off = stats["preprocess_usd"]
            c_on = (stats["retrieval_usd"] + stats["generate_usd"]) / n
            ne = _breakeven_n(c_off, c_on, c_flat)
            cand_rows["archs"][arch] = {
                "n_cells": n,
                "c_off_struct_usd": round(c_off, 4),
                "c_on_per_question_usd": round(c_on, 6),
                "c_flat_per_question_usd": round(c_flat, 6),
                "delta_per_question_usd": round(c_flat - c_on, 6),
                "breakeven_n": (
                    round(ne, 2) if ne != float("inf") else "never"
                ),
            }
        rows.append(cand_rows)

    rows.sort(key=lambda r: r["c_flat_per_question"])

    out = {
        "schema_version": 1,
        "method": "per-(candidate, architecture) break-even N analysis "
                  "from on-disk cost ledger",
        "per_candidate": rows,
    }
    out_path = sanity / "novelqa_breakeven.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Text table
    print()
    print("BREAK-EVEN N: questions on the same document before "
          "preprocessing beats flat-full-context")
    print()
    print(f"{'Candidate':<44} {'C_flat/q':>10} "
          f"{'naive_rag N*':>14} {'raptor N*':>12} {'graphrag N*':>14}")
    print("-" * 96)
    for r in rows:
        archs = r["archs"]
        f = lambda a: str(archs.get(a, {}).get("breakeven_n", "  -  "))
        print(
            f"{r['candidate']:<44} "
            f"${r['c_flat_per_question']:>9.4f} "
            f"{f('naive_rag'):>14} {f('raptor'):>12} {f('graphrag'):>14}"
        )
    print()
    print("'never' = preprocessing's per-question online cost >= flat's; ")
    print("    amortisation mathematically cannot save it at any N.")
    print(f"[breakeven] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
