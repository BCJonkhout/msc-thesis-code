"""Re-parse the 2026-05-19 Phase G rerun RAPTOR + GraphRAG predictions.

Variant of ``novelqa_reparse_predictions_rescored.py`` that targets the
rerun run-dirs at ``outputs/runs/20260519-*`` (RAPTOR + GraphRAG only,
3-novel slice: B01, B08, B50, 15 questions per (cand, arch)) and writes
to ``outputs/rescore_20260519/<run_id>/``.

Same patched MC parser (commit ``b9ce51d``).

The run-id <-> candidate mapping is derived from the rerun log files at
``outputs/phase_g_rerun_2026-05-18/logs/phase2par_<cand>.log`` — each
log emits ``[step3-dry-run] run_id=<rid> items=15 archs=[...]`` exactly
once.

Run from ``code/``::

    .venv/Scripts/python.exe scripts/novelqa_reparse_predictions_rerun_20260519.py

Provenance: PILOT-ERA prediction-repair step for the 2026-05-19 rerun
(re-parses RAPTOR/GraphRAG MC answers with the patched parser, b9ce51d);
feeds the rescored consensus/gold tau-b scripts. Kept as a reproducibility
record; not the source of a current paper claim.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from pilot.sanity.mc_postprocessor import parse_mc_answer


ARCHS = ("raptor", "graphrag")

RUNID_PAT = re.compile(r"\[step3-dry-run\] run_id=([0-9a-f-]+)\s")


def _runid_for_log(log: Path) -> str | None:
    with log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = RUNID_PAT.search(line)
            if m:
                return m.group(1)
    return None


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


def discover_rerun_candidates(logs_dir: Path, runs_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (candidate_label, run_id, run_dir) for every rerun lane."""
    out: list[tuple[str, str, Path]] = []
    for log in sorted(logs_dir.glob("phase2par_*.log")):
        cand = log.stem.replace("phase2par_", "")
        rid = _runid_for_log(log)
        if rid is None:
            continue
        rd = runs_dir / rid
        if rd.exists():
            out.append((cand, rid, rd))
    return out


def main() -> int:
    code_root = Path(__file__).resolve().parents[1]
    logs = code_root / "outputs" / "phase_g_rerun_2026-05-18" / "logs"
    runs = code_root / "outputs" / "runs"
    calib = code_root / "data" / "novelqa" / "calibration_pool.jsonl"
    rescore_root = code_root / "outputs" / "rescore_20260519"
    rescore_root.mkdir(parents=True, exist_ok=True)

    options_by_q = _question_options(calib)
    candidates = discover_rerun_candidates(logs, runs)

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
        "rerun_date": "2026-05-19",
        "scope": "RAPTOR + GraphRAG, 3-novel slice (B01, B08, B50), 15 questions per (cand, arch)",
        "per_candidate_per_arch": {c: dict(d) for c, d in summary.items()},
        "totals": totals,
    }
    impact_path = rescore_root / "parser_impact_summary.json"
    impact_path.write_text(json.dumps(impact, indent=2), encoding="utf-8")

    print(f"[rescore-rerun] {len(candidates)} candidates re-parsed")
    print(f"[rescore-rerun] wrote rescored predictions under {rescore_root}")
    print(f"[rescore-rerun] wrote parser-impact summary to {impact_path}")
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
