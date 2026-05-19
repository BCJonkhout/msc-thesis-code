"""Re-parse Phase G NovelQA predictions with the patched MC parser.

After the 2026-05-16 parser fixes (commit b9ce51d):
  (a) 'S'-corruption from matching the word "options" with the
      case-insensitive keyword pattern is fixed by a trailing word-
      boundary on the keyword group.
  (b) "Letter\\n\\nExplanation" outputs now parse via a dedicated
      pattern that fires before the keyword branch.
  (c) Option-text fallback now uses word-boundary matching so short
      option strings like "No" cannot match inside "not" on yes/no
      questions.

This script does NOT modify any original prediction files. For every
Phase G candidate run it writes a parallel
``<arch>_predictions_rescored.jsonl`` under
``outputs/rescore_20260516/<run_id>/`` containing the original row
fields plus:

  * ``predicted_letter_original``  the value stored in the original
    prediction file (may be None / wrong under the old parser).
  * ``predicted_letter_rescored``  the letter returned by the patched
    parser run against the row's ``predicted_answer`` and the option
    list looked up from the calibration pool.
  * ``parser_changed``             True if rescored != original.
  * ``empty_predicted_answer``     True if ``predicted_answer`` is
    blank/whitespace (these are the cells that must be excluded from
    accuracy, not silently scored as wrong).

Also dumps a parser-impact summary at
``outputs/rescore_20260516/parser_impact_summary.json``.

Run from ``code/``::

    .venv/Scripts/python.exe scripts/novelqa_reparse_predictions_rescored.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pilot.sanity.mc_postprocessor import parse_mc_answer


ARCHS = ("flat", "naive_rag", "raptor", "graphrag")


def _latest_run_id(log: Path) -> str | None:
    rids: list[str] = []
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "run_id=" not in line:
                continue
            rid = line.split("run_id=", 1)[1].split()[0].rstrip(",")
            rids.append(rid)
    return rids[-1] if rids else None


def _question_options(calib_path: Path) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    with calib_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            opts = q.get("Options") or {}
            opt_texts = [opts[k] for k in sorted(opts.keys())]
            out[(q["novel_id"], q["question_id"])] = opt_texts
    return out


def discover_candidates(sanity_dir: Path, runs_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (candidate_label, run_id, run_dir) for every Phase G NovelQA lane."""
    out: list[tuple[str, str, Path]] = []
    for log in sorted(sanity_dir.glob("pareto_novelqa_g_*.log")):
        cand = log.stem.replace("pareto_novelqa_", "")
        rid = _latest_run_id(log)
        if rid is None:
            continue
        rd = runs_dir / rid
        if rd.exists():
            out.append((cand, rid, rd))
    return out


def main() -> int:
    code_root = Path(__file__).resolve().parents[1]
    sanity = code_root / "outputs" / "sanity"
    runs = code_root / "outputs" / "runs"
    calib = code_root / "data" / "novelqa" / "calibration_pool.jsonl"
    rescore_root = code_root / "outputs" / "rescore_20260516"
    rescore_root.mkdir(parents=True, exist_ok=True)

    options_by_q = _question_options(calib)
    candidates = discover_candidates(sanity, runs)

    # Aggregate per-(cand, arch) counters for the summary.
    summary: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {a: {"n_rows": 0, "n_empty": 0, "n_letter_changed": 0,
                     "n_was_S_now_other": 0, "n_was_none_now_letter": 0,
                     "n_was_letter_now_none": 0}
                 for a in ARCHS}
    )

    for cand, rid, rd in candidates:
        out_dir = rescore_root / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        for arch in ARCHS:
            src = rd / f"{arch}_predictions.jsonl"
            if not src.exists():
                continue
            dst = out_dir / f"{arch}_predictions_rescored.jsonl"
            with src.open(encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out_fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("dataset") != "novelqa":
                        out_fh.write(line)
                        continue
                    paper = row["paper_id"]
                    qid = row["question_id"]
                    opts = options_by_q.get((paper, qid)) or []
                    pred_text = row.get("predicted_answer") or ""
                    empty = not pred_text.strip()
                    rescored = None if empty else parse_mc_answer(pred_text, opts)
                    original = row.get("predicted_letter")
                    changed = rescored != original
                    enriched = dict(row)
                    enriched["predicted_letter_original"] = original
                    enriched["predicted_letter_rescored"] = rescored
                    enriched["parser_changed"] = changed
                    enriched["empty_predicted_answer"] = empty
                    out_fh.write(json.dumps(enriched) + "\n")
                    # Counters
                    c = summary[cand][arch]
                    c["n_rows"] += 1
                    if empty:
                        c["n_empty"] += 1
                    if changed:
                        c["n_letter_changed"] += 1
                        if original == "S" and rescored != "S":
                            c["n_was_S_now_other"] += 1
                        if original is None and rescored is not None:
                            c["n_was_none_now_letter"] += 1
                        if original is not None and rescored is None:
                            c["n_was_letter_now_none"] += 1

    # Per-arch and per-candidate totals.
    totals = {
        "by_arch": {a: {"n_rows": 0, "n_empty": 0, "n_letter_changed": 0,
                        "n_was_S_now_other": 0, "n_was_none_now_letter": 0,
                        "n_was_letter_now_none": 0}
                    for a in ARCHS},
        "by_candidate": {},
        "grand": {"n_rows": 0, "n_empty": 0, "n_letter_changed": 0,
                  "n_was_S_now_other": 0, "n_was_none_now_letter": 0,
                  "n_was_letter_now_none": 0},
    }
    for cand, per_arch in summary.items():
        c_row = {"n_rows": 0, "n_empty": 0, "n_letter_changed": 0,
                 "n_was_S_now_other": 0, "n_was_none_now_letter": 0,
                 "n_was_letter_now_none": 0}
        for arch in ARCHS:
            counts = per_arch[arch]
            for k, v in counts.items():
                totals["by_arch"][arch][k] += v
                c_row[k] += v
                totals["grand"][k] += v
        totals["by_candidate"][cand] = c_row

    impact = {
        "schema_version": 1,
        "patched_parser_commit": "b9ce51d",
        "per_candidate_per_arch": {c: dict(d) for c, d in summary.items()},
        "totals": totals,
    }
    impact_path = rescore_root / "parser_impact_summary.json"
    impact_path.write_text(json.dumps(impact, indent=2), encoding="utf-8")

    # Stdout summary
    print(f"[rescore] {len(candidates)} candidates re-parsed")
    print(f"[rescore] wrote rescored predictions under {rescore_root}")
    print(f"[rescore] wrote parser-impact summary to {impact_path}")
    print()
    print("Grand totals:")
    for k, v in totals["grand"].items():
        print(f"  {k}: {v}")
    print()
    print(f"{'candidate':<46} {'rows':>5} {'empty':>5} {'chg':>4} "
          f"{'S->X':>5} {'None->L':>8} {'L->None':>8}")
    for cand in sorted(summary.keys()):
        c = totals["by_candidate"][cand]
        print(f"{cand:<46} {c['n_rows']:>5} {c['n_empty']:>5} "
              f"{c['n_letter_changed']:>4} {c['n_was_S_now_other']:>5} "
              f"{c['n_was_none_now_letter']:>8} {c['n_was_letter_now_none']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
