"""Step 4 variance — across-run F1 stability.

Per pilot plan § 5 Step 4 + § 5.8 row #12, the variance pilot runs
the same architecture × question combination N≥2 times under the
fixed sampling temperature (T=0) and computes per-question variance
to decide N for the main study.

Greedy decoding at the kernel level is not bit-deterministic on
modern provider stacks (batch-invariance breaks, prefix caching,
load balancing across replicas, speculative decoding). The pilot
therefore reports per-question F1 variance under the same prompt /
same architecture / same model and uses the result to set N.

Usage::

    python -m pilot.cli.step_4_variance \\
        --runs outputs/runs/<run1> outputs/runs/<run2> outputs/runs/<run3> \\
        --architecture flat \\
        --metric answer_f1
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def _load_predictions(run_dir: Path, arch: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = run_dir / f"{arch}_predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("paper_id", ""), row.get("question_id", ""))
            out[key] = row
    return out


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = statistics.fmean(xs)
    s = statistics.stdev(xs) if len(xs) >= 2 else 0.0
    return m, s


def compute_variance(
    run_dirs: list[Path], architecture: str, metric: str = "answer_f1"
) -> dict[str, Any]:
    runs = [_load_predictions(rd, architecture) for rd in run_dirs]
    common_keys = sorted(set.intersection(*(set(r) for r in runs)))
    if not common_keys:
        raise RuntimeError("no questions appear in every run dir")

    per_run_macro: list[float] = []
    for r in runs:
        vals = [
            float(r[k][metric]) for k in common_keys
            if isinstance(r[k].get(metric), (int, float))
        ]
        per_run_macro.append(sum(vals) / len(vals) if vals else 0.0)

    macro_mean, macro_sd = _mean_sd(per_run_macro)

    # Per-question across-run SD
    per_q_sds: list[float] = []
    for k in common_keys:
        vals = [
            float(r[k][metric]) for r in runs
            if isinstance(r[k].get(metric), (int, float))
        ]
        if len(vals) >= 2:
            per_q_sds.append(statistics.stdev(vals))

    sd_mean, _ = _mean_sd(per_q_sds)
    sd_max = max(per_q_sds) if per_q_sds else 0.0
    sd_p95 = (
        sorted(per_q_sds)[max(0, math.ceil(0.95 * len(per_q_sds)) - 1)]
        if per_q_sds else 0.0
    )

    # 95% CI half-width on the macro-F1 mean (Student-t approximation
    # at small N would be more honest, but the pilot uses normal-approx
    # to set N; revisit when N ≥ 5 makes t→z trivial).
    n = len(per_run_macro)
    sem = macro_sd / math.sqrt(n) if n >= 2 else 0.0
    ci95_half = 1.96 * sem

    return {
        "architecture": architecture,
        "metric": metric,
        "n_runs": n,
        "n_questions_intersected": len(common_keys),
        "per_run_macro": [round(x, 4) for x in per_run_macro],
        "macro_mean": round(macro_mean, 4),
        "macro_sd": round(macro_sd, 4),
        "sem": round(sem, 4),
        "ci95_half_width": round(ci95_half, 4),
        "per_question_sd_mean": round(sd_mean, 4),
        "per_question_sd_p95": round(sd_p95, 4),
        "per_question_sd_max": round(sd_max, 4),
        "run_dirs": [str(rd) for rd in run_dirs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="Run dirs to compare (one per repeat)")
    parser.add_argument("--architecture", default="flat",
                        choices=["flat", "naive_rag", "raptor", "graphrag"])
    parser.add_argument("--metric", default="answer_f1",
                        choices=["answer_f1", "evidence_f1", "accuracy"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if len(args.runs) < 2:
        print("Need at least 2 run dirs for variance.", file=sys.stderr)
        return 2

    verdict = compute_variance(args.runs, args.architecture, args.metric)
    print(json.dumps(verdict, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
