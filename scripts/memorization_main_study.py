"""Memorization control on the main-study 55-novel pool, per architecture.

Closed-book floor = accuracy with the document withheld (question, and options for
NovelQA). NovelQA's public-domain classics are recalled well above chance, so its
absolute accuracies are memorization-inflated; QASPER's research papers are not,
so its scores reflect genuine reading. The per-architecture "reading lift" (with-
document accuracy minus the closed-book floor) shows how much usable evidence each
architecture actually supplies.

The NovelQA floor is re-scored here over the SAME 55-novel pool the quality numbers
use (calibration novels held out, Frankenstein dropped for missing gold, B42/Les
Mis\'erables recovered via the accent-aware title join), so the floor and the with-
document accuracies are directly comparable. QASPER is unchanged (955 q / 249
papers); its closed-book F1 comes from qasper_nocontext_predictions.jsonl.

Output: code/outputs/main_study/memorization_control.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pilot.env import load_env
from pilot.codabench.extract_score import fetch_correctness_strings, _norm_title

MS = ROOT / "outputs" / "main_study"
DATA = ROOT / "data"
ARCHS = ["flat", "naive_rag", "raptor", "graphrag"]
NOVELQA_NOCONTEXT_SUBMISSION = 799794


def main() -> int:
    load_env()

    # Per-architecture with-document scores + the 55-novel pool, from scored_cells.
    nov = defaultdict(list)
    qa = defaultdict(list)
    incl_nov = set()
    for line in open(MS / "scored_cells.jsonl", encoding="utf-8"):
        o = json.loads(line)
        if o["dataset"] == "novelqa":
            incl_nov.add(o["cluster"]); nov[o["arch"]].append(o["metric"])
        else:
            qa[o["arch"]].append(o["metric"])
    novelqa_acc = {a: statistics.mean(nov[a]) for a in ARCHS}
    qasper_f1 = {a: statistics.mean(qa[a]) for a in ARCHS}
    n_novelqa = len(nov["flat"])
    n_qasper = len(qa["flat"])

    # NovelQA closed-book floor, re-scored over the SAME 55-novel pool.
    nids, qorder = set(), defaultdict(list)
    for line in open(DATA / "novelqa" / "questions.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        q = json.loads(line)
        nids.add(q["novel_id"]); qorder[q["novel_id"]].append(q["question_id"])
    meta = json.loads((DATA / "novelqa" / "bookmeta.json").read_text(encoding="utf-8"))
    norm2nid = {_norm_title(meta[n]["title"]): n for n in nids if n in meta and "title" in meta[n]}
    strings = fetch_correctness_strings(NOVELQA_NOCONTEXT_SUBMISSION)
    floor_vals = []
    for norm, s in strings.items():
        nid = norm2nid.get(_norm_title(norm))
        if not nid or nid not in incl_nov:
            continue
        qids = qorder.get(nid, [])
        for i, ch in enumerate(s):
            if i < len(qids):
                floor_vals.append(1 if ch == "T" else 0)
    novelqa_floor = statistics.mean(floor_vals)

    # QASPER closed-book F1 (document withheld), unchanged pool.
    qa_floor_vals = [o["answer_f1"] for o in
                     (json.loads(l) for l in open(MS / "qasper_nocontext_predictions.jsonl", encoding="utf-8"))
                     if isinstance(o.get("answer_f1"), (int, float))]
    qasper_floor = statistics.mean(qa_floor_vals)

    out = {
        # backward-compatible top-level keys (NovelQA flat headline)
        "nocontext_accuracy": round(novelqa_floor, 4),
        "flat_accuracy": round(novelqa_acc["flat"], 4),
        "lift": round(novelqa_acc["flat"] - novelqa_floor, 4),
        "novelqa": {
            "closed_book": round(novelqa_floor, 4),
            "n_questions": len(floor_vals),
            "submission_id": NOVELQA_NOCONTEXT_SUBMISSION,
            "per_arch": {a: {"accuracy": round(novelqa_acc[a], 4),
                             "lift": round(novelqa_acc[a] - novelqa_floor, 4)} for a in ARCHS},
        },
        "qasper": {
            "closed_book": round(qasper_floor, 4),
            "nocontext_answer_f1": round(qasper_floor, 4),
            "flat_answer_f1": round(qasper_f1["flat"], 4),
            "lift": round(qasper_f1["flat"] - qasper_floor, 4),
            "n_questions": n_qasper,
            "per_arch": {a: {"answer_f1": round(qasper_f1[a], 4),
                             "lift": round(qasper_f1[a] - qasper_floor, 4)} for a in ARCHS},
        },
    }
    (MS / "memorization_control.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"NovelQA closed-book floor (55-pool, n={len(floor_vals)}): {novelqa_floor:.4f}")
    for a in ARCHS:
        print(f"  {a:10s} NovelQA {novelqa_acc[a]:.4f} (lift {novelqa_acc[a]-novelqa_floor:+.4f})  "
              f"QASPER {qasper_f1[a]:.4f} (lift {qasper_f1[a]-qasper_floor:+.4f})")
    print(f"QASPER closed-book floor (n={n_qasper}): {qasper_floor:.4f}")
    print(f"wrote {MS/'memorization_control.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
