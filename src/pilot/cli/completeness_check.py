"""Completeness gate for a run directory.

A cross-architecture comparison is only valid on a COMPLETE grid: every
(architecture, paper, question, run_index) cell must be present, or the
Kendall / accuracy / variance scorers silently average a ragged sample
and the ranking is biased (the same reason B41 was dropped symmetrically
in the pilot). This module computes the expected cell set from the same
loaders the sweep uses and reports the missing cells, exiting non-zero
when the run is incomplete.

Operational use: run this after a sweep and BEFORE scoring. On a
non-empty miss list, re-invoke the sweep with the same config — resume-
in-place fills exactly the gaps (idempotent) — then re-check. Loop until
clean. This converts a "silently dropped cell" into an explicit,
re-drivable gap.

    python -m pilot.cli.completeness_check \\
        --run-dir outputs/runs/<run_id> \\
        --architectures flat naive_rag raptor graphrag \\
        --datasets qasper novelqa --num-runs 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pilot.cli.step_3_dry_run import (
    _project_root,
    load_novelqa_calibration,
    load_novelqa_full,
    load_qasper_calibration,
    load_qasper_full,
)


def expected_cells(
    *,
    data_root: Path,
    datasets: list[str],
    architectures: list[str],
    num_runs: int,
    split: str = "calibration",
) -> set[tuple[str, str, str, int]]:
    """The full grid of (arch, paper_id, question_id, run_index) cells.

    ``split`` MUST match the split the run actually used: a 'full' run is
    validated against the full loaders, a 'calibration' run against the
    20+20 calibration pools. Validating a full run against the calibration
    set would mark it COMPLETE while leaving the real grid unchecked.
    """
    items: list[dict[str, Any]] = []
    if "qasper" in datasets:
        items.extend(
            load_qasper_full(data_root) if split == "full"
            else load_qasper_calibration(data_root)
        )
    if "novelqa" in datasets:
        items.extend(
            load_novelqa_full(data_root) if split == "full"
            else load_novelqa_calibration(data_root)
        )
    cells: set[tuple[str, str, str, int]] = set()
    for it in items:
        for arch in architectures:
            for run_index in range(num_runs):
                cells.add((arch, it["paper_id"], it["question_id"], run_index))
    return cells


def present_cells(
    run_dir: Path, architectures: list[str]
) -> set[tuple[str, str, str, int]]:
    """The (arch, paper_id, question_id, run_index) cells already on disk."""
    cells: set[tuple[str, str, str, int]] = set()
    for arch in architectures:
        path = run_dir / f"{arch}_predictions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a torn trailing line; it will be re-run.
                continue
            cells.add((
                arch,
                row.get("paper_id", ""),
                row.get("question_id", ""),
                int(row.get("run_index", 0)),
            ))
    return cells


def missing_cells(
    *,
    run_dir: Path,
    data_root: Path,
    datasets: list[str],
    architectures: list[str],
    num_runs: int,
    split: str = "calibration",
) -> set[tuple[str, str, str, int]]:
    """Expected cells that are not present on disk."""
    expected = expected_cells(
        data_root=data_root, datasets=datasets,
        architectures=architectures, num_runs=num_runs, split=split,
    )
    return expected - present_cells(run_dir, architectures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--architectures", nargs="+",
        default=["flat", "naive_rag", "raptor", "graphrag"],
    )
    parser.add_argument("--datasets", nargs="+", default=["qasper", "novelqa"])
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=_project_root() / "data")
    parser.add_argument(
        "--split", choices=["calibration", "full"], default=None,
        help="evaluation split the run used; if omitted, read from "
             "run_manifest.json (fallback: calibration).",
    )
    args = parser.parse_args()

    split = args.split
    if split is None:
        manifest = args.run_dir / "run_manifest.json"
        if manifest.exists():
            try:
                split = json.loads(manifest.read_text(encoding="utf-8")).get("split")
            except (json.JSONDecodeError, OSError):
                split = None
        split = split or "calibration"

    missing = missing_cells(
        run_dir=args.run_dir,
        data_root=args.data_root,
        datasets=args.datasets,
        architectures=args.architectures,
        num_runs=args.num_runs,
        split=split,
    )
    if not missing:
        print(f"[completeness] COMPLETE: {args.run_dir} has every expected cell")
        return 0

    per_arch = Counter(c[0] for c in missing)
    print(
        f"[completeness] INCOMPLETE: {len(missing)} missing cells in {args.run_dir}",
        file=sys.stderr,
    )
    for arch in sorted(per_arch):
        print(f"  {arch}: {per_arch[arch]} missing", file=sys.stderr)
    # A small sample so the operator can eyeball which cells.
    for cell in sorted(missing)[:20]:
        print(f"    MISSING {cell[0]}/{cell[1]}/{cell[2]}#r{cell[3]}", file=sys.stderr)
    if len(missing) > 20:
        print(f"    ... and {len(missing) - 20} more", file=sys.stderr)
    print(
        "[completeness] re-run the sweep with the same config to fill the "
        "gaps (resume-in-place), then re-check before scoring.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
