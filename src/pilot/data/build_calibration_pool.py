"""Calibration-pool builder.

Per pilot plan § 2.1 (QASPER) and § 2.2 (NovelQA), the calibration
pool is 20 + 20 questions used for the pre-main-study calibration
dry run (Step 3), variance measurement (Step 4), and encoder /
summary-model selection sub-experiments. The pool is fixed by a
deterministic seed so re-running the builder produces the same
pool across machines and across time.

QASPER pool rules (§ 2.1):
  - 20 questions from the dev split.
  - Each from a distinct paper.
  - Each must have at least one annotated evidence sentence (so
    Evidence-F1 is computable without the main study).
  - Reject any paper that also appears in the test split (assert).

NovelQA pool rules (§ 2.2):
  - 20 multiple-choice questions, each from a distinct novel.
  - Drawn from a 4-novel held-aside calibration set (4 × 5).
  - Calibration novels are excluded from the held-out main
    evaluation — recorded explicitly in the output.

Output:
  data/qasper/calibration_pool.jsonl        — 20 (paper_id, question_id, ...)
  data/novelqa/calibration_pool.jsonl       — 20 (novel_id, question_id, ...)
  data/novelqa/calibration_novels.json      — 4-novel exclusion list
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pilot.env import load_env


_DEFAULT_SEED = 42
_QASPER_POOL_SIZE = 20
_NOVELQA_POOL_SIZE = 20
_NOVELQA_CAL_NOVELS = 4
_NOVELQA_QUESTIONS_PER_NOVEL = 5

# B48 (The History of Rome, 2.58M tokens) is excluded from main eval
# per pilot plan § 2.2 Exclusion. Skip it from calibration sampling
# too — it's an outlier stress test, not a calibration row.
_NOVELQA_EXCLUDE = {"B48"}


def _project_data_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3] / "data"


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _has_evidence(answer: dict[str, Any]) -> bool:
    """Per QASPER schema, an answer carries an 'evidence' list of
    paragraph-level snippets and a 'highlighted_evidence' list of
    sentence-level spans. Either non-empty qualifies as 'has annotated
    evidence' for our calibration filter."""
    a = answer.get("answer", {})
    if not isinstance(a, dict):
        return False
    return bool(a.get("evidence") or a.get("highlighted_evidence"))


def build_qasper_pool(data_root: Path, seed: int) -> dict[str, Any]:
    """Sample 20 evidence-bearing questions from distinct dev-split papers."""
    dev_path = data_root / "qasper" / "dev.jsonl"
    test_path = data_root / "qasper" / "test.jsonl"
    if not dev_path.exists():
        return {"dataset": "qasper", "status": "error", "error": f"missing {dev_path}"}

    dev = _load_jsonl(dev_path)
    test = _load_jsonl(test_path) if test_path.exists() else []
    test_paper_ids = {p["paper_id"] for p in test}

    # Build a candidate list of (paper_id, question_id, question_text)
    # tuples filtered to those with at least one evidence-bearing answer
    # AND whose paper does not also appear in the test split.
    candidates_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in dev:
        paper_id = paper["paper_id"]
        if paper_id in test_paper_ids:
            continue
        for qa in paper.get("qas", []):
            if not any(_has_evidence(a) for a in qa.get("answers", [])):
                continue
            candidates_by_paper[paper_id].append(
                {
                    "paper_id": paper_id,
                    "question_id": qa["question_id"],
                    "question": qa["question"],
                    "title": paper.get("title"),
                }
            )

    # One question per paper, then sample 20 papers.
    rng = random.Random(seed)
    paper_ids = sorted(candidates_by_paper.keys())
    rng.shuffle(paper_ids)

    pool: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        if len(pool) >= _QASPER_POOL_SIZE:
            break
        # Deterministically pick the first evidence-bearing question of this paper
        # (sorted by question_id for stability).
        candidates = sorted(candidates_by_paper[paper_id], key=lambda r: r["question_id"])
        pool.append(candidates[0])

    if len(pool) < _QASPER_POOL_SIZE:
        return {
            "dataset": "qasper",
            "status": "error",
            "error": (
                f"only {len(pool)} evidence-bearing dev papers found; "
                f"need {_QASPER_POOL_SIZE}"
            ),
        }

    out_path = data_root / "qasper" / "calibration_pool.jsonl"
    _atomic_write_jsonl(out_path, pool)

    return {
        "dataset": "qasper",
        "status": "built",
        "path": str(out_path),
        "pool_size": len(pool),
        "candidate_papers": len(candidates_by_paper),
        "seed": seed,
    }


def build_novelqa_pool(data_root: Path, seed: int) -> dict[str, Any]:
    """Pick 4 calibration novels, sample 5 questions per novel.

    Calibration novels are recorded in calibration_novels.json so the
    held-out main evaluation can exclude them. B48 is structurally
    excluded (outlier stress test, not calibration material).
    """
    questions_path = data_root / "novelqa" / "questions.jsonl"
    full_texts_dir = data_root / "novelqa" / "full_texts"
    if not questions_path.exists():
        return {"dataset": "novelqa", "status": "error", "error": f"missing {questions_path}"}

    available_novels = sorted(
        {p.stem for p in full_texts_dir.glob("*.txt")} - _NOVELQA_EXCLUDE
    )

    questions = _load_jsonl(questions_path)
    by_novel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        novel_id = q["novel_id"]
        if novel_id in _NOVELQA_EXCLUDE:
            continue
        if novel_id not in set(available_novels):
            # Question references a novel whose text isn't in the
            # public-domain set; skip — pilot can only evaluate
            # end-to-end where the text is available.
            continue
        by_novel[novel_id].append(q)

    eligible_novels = [
        n for n in available_novels
        if len(by_novel.get(n, [])) >= _NOVELQA_QUESTIONS_PER_NOVEL
    ]
    if len(eligible_novels) < _NOVELQA_CAL_NOVELS:
        return {
            "dataset": "novelqa",
            "status": "error",
            "error": (
                f"only {len(eligible_novels)} novels have ≥ "
                f"{_NOVELQA_QUESTIONS_PER_NOVEL} questions; "
                f"need {_NOVELQA_CAL_NOVELS}"
            ),
        }

    rng = random.Random(seed)
    cal_novels = rng.sample(eligible_novels, _NOVELQA_CAL_NOVELS)
    cal_novels.sort()

    pool: list[dict[str, Any]] = []
    for novel_id in cal_novels:
        candidates = sorted(by_novel[novel_id], key=lambda r: r["question_id"])
        # Deterministic 5-question sample per novel via seeded RNG.
        local_rng = random.Random(int(hashlib.sha256(f"{seed}-{novel_id}".encode()).hexdigest(), 16) % (2**32))
        sampled = local_rng.sample(candidates, _NOVELQA_QUESTIONS_PER_NOVEL)
        pool.extend(sampled)

    out_pool = data_root / "novelqa" / "calibration_pool.jsonl"
    out_novels = data_root / "novelqa" / "calibration_novels.json"
    _atomic_write_jsonl(out_pool, pool)
    out_novels.write_text(
        json.dumps(
            {
                "calibration_novels": cal_novels,
                "exclude_from_main_eval": cal_novels + sorted(_NOVELQA_EXCLUDE),
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "dataset": "novelqa",
        "status": "built",
        "path": str(out_pool),
        "calibration_novels": cal_novels,
        "pool_size": len(pool),
        "available_novels": len(available_novels),
        "eligible_novels": len(eligible_novels),
        "seed": seed,
    }


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser(
        description="Build deterministic calibration pools for QASPER + NovelQA."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_project_data_root(),
        help="Data root directory. Default: code/data/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"RNG seed (default {_DEFAULT_SEED}).",
    )
    args = parser.parse_args()

    results = [
        build_qasper_pool(args.data_root, args.seed),
        build_novelqa_pool(args.data_root, args.seed),
    ]
    for r in results:
        print(json.dumps(r, indent=2))

    return 0 if all(r["status"] == "built" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
