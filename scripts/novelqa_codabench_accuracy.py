"""Recover this study's NovelQA accuracy from a Codabench submission.

The leaderboard's aggregate accuracy is diluted by the ~29 copyright-protected
novels we cannot run end-to-end (filled with the placeholder letter). The real
number is over the public-domain novels we actually predicted. The platform's
scoring_stdout carries per-novel T/F correctness strings, so we slice those to
our novels and report accuracy over exactly the questions we answered.

Usage:
    python scripts/novelqa_codabench_accuracy.py --submission-id <id> [--label flat]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.env import load_env
from pilot.codabench.extract_score import fetch_correctness_strings, _norm_title


def _data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _our_novels() -> dict[str, str]:
    """novel_id -> title for the public-domain novels we predicted."""
    dr = _data_root()
    ids = set()
    with (dr / "novelqa" / "questions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ids.add(json.loads(line)["novel_id"])
    meta = json.loads((dr / "novelqa" / "bookmeta.json").read_text(encoding="utf-8"))
    return {nid: meta[nid]["title"] for nid in ids if nid in meta and "title" in meta[nid]}


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--submission-id", type=int, required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    strings = fetch_correctness_strings(args.submission_id)
    if not strings:
        print(json.dumps({"submission_id": args.submission_id,
                          "error": "no T/F strings recovered from scoring_stdout"}, indent=2))
        return 1

    ours = _our_novels()
    matched = missing = total_t = total_q = 0
    per_novel = {}
    for nid, title in ours.items():
        s = strings.get(_norm_title(title))
        if not s:
            missing += 1
            continue
        matched += 1
        t = s.count("T")
        total_t += t
        total_q += len(s)
        per_novel[nid] = {"title": title, "correct": t, "total": len(s),
                          "accuracy": round(t / len(s), 4) if s else None}

    out = {
        "submission_id": args.submission_id,
        "label": args.label,
        "our_novels_matched": matched,
        "our_novels_missing_in_log": missing,
        "questions_scored": total_q,
        "correct": total_t,
        "accuracy_on_our_novels": round(total_t / total_q, 4) if total_q else None,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
