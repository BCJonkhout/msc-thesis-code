"""Step 3 sub-experiment: encoder Recall@k on QASPER calibration pool.

Per pilot plan § 5.8 row #9, the embedding encoder is selected by
measuring Recall@k against QASPER gold evidence spans on the 20-
question QASPER calibration pool, k ∈ {5, 10, 20}.

Decision rule:
  Recall@20 ≥ 0.85               → BGE-large locked.
  0.70 ≤ Recall@20 < 0.85         → require e5-mistral check; pick
                                    e5 if e5 ≥ BGE + 0.05, else BGE.
  Recall@20 < 0.70                → run e5 + voyage-3-large; pick
                                    highest.
  Recall@5  < 0.50 (any encoder)  → STOP; chunking is broken.

Pipeline (deterministic given the same encoder + chunking + pool):
  1. Load the QASPER calibration pool (20 questions).
  2. For each question, load the parent paper's full text from
     data/qasper/dev.jsonl, linearise the section/paragraph tree
     to plain text, and chunk per methods.yaml#naive_rag (384 tokens,
     0 overlap, sentence-boundary-aware).
  3. Embed all chunks + the question via the configured encoder
     (default Ollama bge-m3).
  4. Cosine-rank the chunks; collect top-k for k ∈ {5, 10, 20}.
  5. Compute recall: a gold-evidence sentence is "hit" if any
     retrieved chunk contains it (case-insensitive substring after
     whitespace normalisation). recall_q = hits / total_gold_sentences.
  6. Macro-average across questions; emit verdict JSON with
     per-question rows + thresholds applied.

Usage:
  python -m pilot.cli.step_3_encoder_recall                  # default bge-m3
  python -m pilot.cli.step_3_encoder_recall --model bge-m3
  python -m pilot.cli.step_3_encoder_recall --k 5 10 20

Verdict file: outputs/sanity/step_3_encoder_recall_<ts>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pilot.encoders import OllamaEmbedder, SentenceBoundaryChunker, TextChunk
from pilot.env import load_env


# Decision thresholds per pilot plan § 5.8 row #9.
_RECALL_AT_20_LOCK = 0.85
_RECALL_AT_20_BORDERLINE = 0.70
_RECALL_AT_5_FLOOR = 0.50

_DEFAULT_MODEL = "bge-m3"
_DEFAULT_K = (5, 10, 20)


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _linearise_paper(paper: dict[str, Any]) -> str:
    """Concatenate a QASPER paper's full_text into one string.

    QASPER's `full_text` is a list of section objects; each carries a
    `section_name` and `paragraphs` (list of strings). This produces
    a plain-text linearisation suitable for chunking. Section names
    are included as headers so paragraph context is not lost.
    """
    parts: list[str] = []
    if paper.get("title"):
        parts.append(str(paper["title"]))
    if paper.get("abstract"):
        parts.append(str(paper["abstract"]))
    for section in paper.get("full_text", []) or []:
        name = section.get("section_name") or ""
        if name:
            parts.append(name)
        for para in section.get("paragraphs", []) or []:
            text = para if isinstance(para, str) else str(para)
            if text.strip():
                parts.append(text)
    return "\n\n".join(parts)


def _gold_evidence(question_record: dict[str, Any]) -> list[str]:
    """Collect all annotated evidence sentences for a QASPER question.

    QASPER stores up to N answer-annotator records per question; each
    annotator carries `evidence` (paragraph snippets) and
    `highlighted_evidence` (sentence-level spans). We use the union
    of `highlighted_evidence` across annotators (sentence-level is
    the precise unit to match against retrieved chunks). Fall back
    to `evidence` if no `highlighted_evidence` is present.
    """
    sentences: list[str] = []
    for ans in question_record.get("answers", []) or []:
        a = ans.get("answer", {})
        if not isinstance(a, dict):
            continue
        for s in a.get("highlighted_evidence", []) or []:
            if isinstance(s, str) and s.strip():
                sentences.append(s.strip())
        # If highlighted is empty for this annotator but evidence is
        # not, fall back. Different annotators can contribute different
        # forms of evidence.
        if not a.get("highlighted_evidence"):
            for s in a.get("evidence", []) or []:
                if isinstance(s, str) and s.strip():
                    sentences.append(s.strip())
    # Deduplicate, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ──────────────────────────────────────────────────────────────────────
# Recall computation
# ──────────────────────────────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")


def _normalise(s: str) -> str:
    return _WS_RE.sub(" ", s.strip().lower())


def _evidence_hit(evidence_sentence: str, chunk_text: str) -> bool:
    """A gold sentence counts as hit if it appears (substring) in the
    retrieved chunk after whitespace + case normalisation, OR if a
    high-overlap word-token match exists. The substring path catches
    near-verbatim matches; the token-overlap path tolerates minor
    tokenizer/whitespace drift between the QASPER annotation and the
    `paragraphs` text."""
    ev = _normalise(evidence_sentence)
    ck = _normalise(chunk_text)
    if not ev or not ck:
        return False
    if ev in ck:
        return True
    # Token-overlap fallback for near-verbatim sentences whose
    # whitespace was slightly mangled.
    ev_tokens = set(ev.split())
    if len(ev_tokens) < 6:
        return False  # too short to be reliable as an overlap signal
    ck_tokens = set(ck.split())
    overlap = len(ev_tokens & ck_tokens) / len(ev_tokens)
    return overlap >= 0.85


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def _topk_indices(scores: list[float], k: int) -> list[int]:
    """Indices of the top-k highest scores, ties broken by earlier
    index so the function is deterministic."""
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: (-x[1], x[0]))
    return [i for i, _ in indexed[:k]]


# ──────────────────────────────────────────────────────────────────────
# Per-question evaluation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class QuestionRecallResult:
    paper_id: str
    question_id: str
    question: str
    chunks_count: int
    gold_evidence_count: int
    recall_at_k: dict[int, float] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None


def _evaluate_question(
    *,
    question_record: dict[str, Any],
    paper_text: str,
    embedder: OllamaEmbedder,
    chunker: SentenceBoundaryChunker,
    k_values: tuple[int, ...],
) -> QuestionRecallResult:
    paper_id = question_record["paper_id"]
    question_id = question_record["question_id"]
    question_text = question_record.get("question", "")

    chunks = chunker.chunk(paper_text)
    gold = question_record["_gold_evidence_sentences"]

    if not chunks or not gold:
        return QuestionRecallResult(
            paper_id=paper_id,
            question_id=question_id,
            question=question_text,
            chunks_count=len(chunks),
            gold_evidence_count=len(gold),
            recall_at_k={k: 0.0 for k in k_values},
            skipped=True,
            skip_reason="no_chunks" if not chunks else "no_gold_evidence",
        )

    chunk_texts = [c.text for c in chunks]
    chunk_vecs = embedder.embed(chunk_texts).embeddings
    question_vec = embedder.embed([question_text]).embeddings[0]

    scores = [_cosine(question_vec, vec) for vec in chunk_vecs]

    recalls: dict[int, float] = {}
    for k in k_values:
        topk = _topk_indices(scores, k)
        retrieved_texts = [chunks[i].text for i in topk]
        hits = 0
        for ev in gold:
            if any(_evidence_hit(ev, t) for t in retrieved_texts):
                hits += 1
        recalls[k] = hits / len(gold)

    return QuestionRecallResult(
        paper_id=paper_id,
        question_id=question_id,
        question=question_text,
        chunks_count=len(chunks),
        gold_evidence_count=len(gold),
        recall_at_k=recalls,
    )


# ──────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────

def _verdict_for(macro_recall_at_20: float, macro_recall_at_5: float) -> dict[str, Any]:
    if macro_recall_at_5 < _RECALL_AT_5_FLOOR:
        return {
            "decision": "STOP_CHUNKING_BROKEN",
            "rationale": (
                f"Recall@5 = {macro_recall_at_5:.3f} < {_RECALL_AT_5_FLOOR:.2f}; "
                "the chunking pipeline cannot retrieve evidence at the smallest k. "
                "Investigate chunk size, overlap, or sentence-splitting before "
                "running any answer-quality measurement."
            ),
        }
    if macro_recall_at_20 >= _RECALL_AT_20_LOCK:
        return {
            "decision": "LOCK_BGE_M3",
            "rationale": (
                f"Recall@20 = {macro_recall_at_20:.3f} ≥ {_RECALL_AT_20_LOCK:.2f}; "
                "BGE-M3 meets the pilot's encoder-lock threshold. No escalation "
                "candidate (e5-mistral, voyage-3-large) needs to run."
            ),
        }
    if macro_recall_at_20 >= _RECALL_AT_20_BORDERLINE:
        return {
            "decision": "ESCALATE_TO_E5_MISTRAL",
            "rationale": (
                f"Recall@20 = {macro_recall_at_20:.3f} in "
                f"[{_RECALL_AT_20_BORDERLINE:.2f}, {_RECALL_AT_20_LOCK:.2f}); "
                "run e5-mistral on the same calibration pool. Pick e5-mistral "
                "if its Recall@20 is at least 0.05 above BGE-M3, else keep BGE-M3."
            ),
        }
    return {
        "decision": "ESCALATE_TO_E5_AND_VOYAGE",
        "rationale": (
            f"Recall@20 = {macro_recall_at_20:.3f} < "
            f"{_RECALL_AT_20_BORDERLINE:.2f}; run both e5-mistral and "
            "voyage-3-large on the same pool and pick the highest Recall@20."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────

def run_step_3_encoder_recall(
    *,
    encoder_model: str,
    k_values: tuple[int, ...],
    data_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    qasper_dev_path = data_root / "qasper" / "dev.jsonl"
    pool_path = data_root / "qasper" / "calibration_pool.jsonl"
    if not qasper_dev_path.exists():
        raise SystemExit(f"missing {qasper_dev_path}; run `make pilot-data-download`")
    if not pool_path.exists():
        raise SystemExit(f"missing {pool_path}; run `make pilot-build-calibration`")

    pool = _load_jsonl(pool_path)
    paper_index = {p["paper_id"]: p for p in _load_jsonl(qasper_dev_path)}

    chunker = SentenceBoundaryChunker(chunk_size_tokens=384, overlap_tokens=0)
    embedder = OllamaEmbedder(model=encoder_model)

    per_question: list[QuestionRecallResult] = []
    macro_sum: dict[int, float] = defaultdict(float)
    macro_count: dict[int, int] = defaultdict(int)

    for entry in pool:
        paper_id = entry["paper_id"]
        question_id = entry["question_id"]
        paper = paper_index.get(paper_id)
        if paper is None:
            per_question.append(
                QuestionRecallResult(
                    paper_id=paper_id,
                    question_id=question_id,
                    question=entry.get("question", ""),
                    chunks_count=0,
                    gold_evidence_count=0,
                    recall_at_k={k: 0.0 for k in k_values},
                    skipped=True,
                    skip_reason="paper_not_in_dev_split",
                )
            )
            continue

        # Find this specific question's full record (with `answers`).
        qa_record = next(
            (qa for qa in paper.get("qas", []) if qa.get("question_id") == question_id),
            None,
        )
        if qa_record is None:
            per_question.append(
                QuestionRecallResult(
                    paper_id=paper_id,
                    question_id=question_id,
                    question=entry.get("question", ""),
                    chunks_count=0,
                    gold_evidence_count=0,
                    recall_at_k={k: 0.0 for k in k_values},
                    skipped=True,
                    skip_reason="question_not_in_paper",
                )
            )
            continue

        qa_record_with_meta = dict(qa_record)
        qa_record_with_meta["paper_id"] = paper_id
        qa_record_with_meta["_gold_evidence_sentences"] = _gold_evidence(qa_record)

        result = _evaluate_question(
            question_record=qa_record_with_meta,
            paper_text=_linearise_paper(paper),
            embedder=embedder,
            chunker=chunker,
            k_values=k_values,
        )
        per_question.append(result)
        if not result.skipped:
            for k in k_values:
                macro_sum[k] += result.recall_at_k[k]
                macro_count[k] += 1

    embedder.close()

    macro_recall: dict[str, float] = {
        f"recall_at_{k}": (macro_sum[k] / macro_count[k]) if macro_count[k] else 0.0
        for k in k_values
    }
    verdict = _verdict_for(
        macro_recall_at_20=macro_recall.get("recall_at_20", 0.0),
        macro_recall_at_5=macro_recall.get("recall_at_5", 0.0),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_3_encoder_recall_{timestamp}.json"
    summary = {
        "timestamp_utc": timestamp,
        "encoder_model": encoder_model,
        "k_values": list(k_values),
        "pool_path": str(pool_path),
        "questions_evaluated": sum(1 for r in per_question if not r.skipped),
        "questions_skipped": sum(1 for r in per_question if r.skipped),
        "macro_recall": macro_recall,
        "verdict": verdict,
        "per_question": [asdict(r) for r in per_question],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["verdict_path"] = str(out_path)
    return summary


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Ollama model id (default {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=list(_DEFAULT_K),
        help=f"k values to compute Recall@k for (default {' '.join(str(x) for x in _DEFAULT_K)}).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_project_root() / "data",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_project_root() / "outputs" / "sanity",
    )
    args = parser.parse_args()

    summary = run_step_3_encoder_recall(
        encoder_model=args.model,
        k_values=tuple(args.k),
        data_root=args.data_root,
        out_dir=args.out,
    )

    print(json.dumps(summary, indent=2))
    print(f"\nWrote verdict: {summary['verdict_path']}")
    decision = summary["verdict"]["decision"]
    return 0 if decision in {"LOCK_BGE_M3"} else 1


if __name__ == "__main__":
    sys.exit(main())
