"""Aggregate Step 3 dry-run predictions into a per-architecture report.

Reads the per-arch ``<arch>_predictions.jsonl`` files written by
``pilot.cli.step_3_dry_run`` (or the verdict JSON it emits) and
prints macro Answer-F1 / Evidence-F1 (QASPER) plus a NovelQA
prediction-coverage summary.

Useful for:
  - inspecting partial results during a long run (predictions are
    flushed incrementally, so this works on an in-progress dir),
  - re-aggregating when the dry-run summary JSON wasn't reached
    because of a crash.

Usage::

    python -m pilot.cli.step_3_summary --run outputs/runs/<run_id>
    python -m pilot.cli.step_3_summary --verdict outputs/sanity/step_3_dry_run_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def aggregate_run_dir(run_dir: Path) -> dict[str, Any]:
    """Walk a run dir's per-arch predictions JSONLs and macro-aggregate."""
    arch_files = sorted(run_dir.glob("*_predictions.jsonl"))
    per_arch: dict[str, dict[str, Any]] = {}

    for f in arch_files:
        arch = f.stem.replace("_predictions", "")
        rows = _load_jsonl(f)
        qasper = [r for r in rows if r.get("dataset") == "qasper"]
        novelqa = [r for r in rows if r.get("dataset") == "novelqa"]

        af = [
            r["answer_f1"] for r in qasper
            if isinstance(r.get("answer_f1"), (int, float))
        ]
        ef = [
            r["evidence_f1"] for r in qasper
            if isinstance(r.get("evidence_f1"), (int, float))
        ]
        novel_letters = [
            r.get("predicted_letter") for r in novelqa
        ]
        parsed_letters = [l for l in novel_letters if l in {"A", "B", "C", "D"}]

        per_arch[arch] = {
            "qasper_count": len(qasper),
            "novelqa_count": len(novelqa),
            "macro_answer_f1_qasper": (sum(af) / len(af)) if af else None,
            "macro_evidence_f1_qasper": (sum(ef) / len(ef)) if ef else None,
            "novelqa_letters_parsed": len(parsed_letters),
            "novelqa_letters_unparsed": len(novel_letters) - len(parsed_letters),
            "novelqa_letter_distribution": _letter_distribution(parsed_letters),
        }

    return {
        "run_dir": str(run_dir),
        "architectures_present": list(per_arch.keys()),
        "per_arch": per_arch,
    }


def _letter_distribution(letters: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for l in letters:
        counts[l] += 1
    return dict(sorted(counts.items()))


def aggregate_verdict_json(path: Path) -> dict[str, Any]:
    """Pretty-print the per_arch_macro block from a verdict JSON."""
    body = json.loads(path.read_text(encoding="utf-8"))
    return {
        "run_id": body.get("run_id"),
        "items_count": body.get("items_count"),
        "per_arch_macro": body.get("per_arch_macro"),
        "failures_count": body.get("failures_count"),
        "ledger_path": body.get("ledger_path"),
    }


def _format_table(summary: dict[str, Any]) -> str:
    """Tab-separated table for terminal display."""
    if "per_arch" not in summary:
        return json.dumps(summary, indent=2)
    headers = [
        "arch", "qasper_n", "novelqa_n",
        "qasper_aF1", "qasper_eF1",
        "novel_parsed", "novel_letters",
    ]
    rows = [headers]
    for arch, stats in summary["per_arch"].items():
        rows.append([
            arch,
            str(stats["qasper_count"]),
            str(stats["novelqa_count"]),
            f"{stats['macro_answer_f1_qasper']:.3f}" if stats["macro_answer_f1_qasper"] is not None else "—",
            f"{stats['macro_evidence_f1_qasper']:.3f}" if stats["macro_evidence_f1_qasper"] is not None else "—",
            f"{stats['novelqa_letters_parsed']}/{stats['novelqa_count']}",
            ",".join(f"{k}={v}" for k, v in stats["novelqa_letter_distribution"].items()),
        ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    out = []
    for r in rows:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(r))))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run", type=Path, help="Run directory containing *_predictions.jsonl")
    grp.add_argument("--verdict", type=Path, help="Verdict JSON from a closed dry run")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args()

    if args.run:
        summary = aggregate_run_dir(args.run)
    else:
        summary = aggregate_verdict_json(args.verdict)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(_format_table(summary))
        if "run_dir" in summary:
            print(f"\nrun_dir: {summary['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
