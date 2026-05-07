"""Step 3 calibration-pool dry run.

Per pilot plan § 5 Step 3, this is the first end-to-end run of the
benchmark architectures on real data. Each architecture is invoked
once per (paper_or_novel, question) at N=1; predictions and ledger
rows are written to disk; scoring runs immediately for QASPER (which
ships local labels) and is deferred for NovelQA (whose labels are
held by the leaderboard at novelqa.github.io).

Predictions land at:
  outputs/runs/<run_id>/<architecture>_predictions.jsonl
Ledger rows land at:
  outputs/runs/<run_id>/ledger.jsonl
Verdict summary:
  outputs/sanity/step_3_dry_run_<ts>.json

Architectures supported in this iteration:
  - flat       (full-context, no retrieval)
  - naive_rag  (chunk + embed + top-k cosine + answer)

RAPTOR and GraphRAG land in subsequent iterations as they require
heavier external dependencies (parthsarthi03/raptor + microsoft/graphrag).

Default answerer model: gemini-3.1-pro-preview (the natural primary
closed candidate post-Step-2; cache primitive verified). Override via
``--answerer-model``.

Usage:
    python -m pilot.cli.step_3_dry_run \\
        --architectures flat naive_rag \\
        --datasets qasper novelqa \\
        --answerer-model gemini-3.1-pro-preview
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pilot.architectures import ArchitectureResult, run_flat, run_naive_rag
from pilot.encoders import OllamaEmbedder, SentenceBoundaryChunker
from pilot.env import load_env
from pilot.eval import (
    answer_f1_against_references,
    accuracy,
    evidence_f1,
    parse_mc_answer,
)
from pilot.ledger import CostLedger, new_run_id
from pilot.providers import get_provider
from pilot.providers.base import CacheControl


_DEFAULT_ANSWERER_MODEL = "gemini-3.1-pro-preview"
_DEFAULT_ANSWERER_PROVIDER = "gemini"
_DEFAULT_EMBEDDER_MODEL = "bge-m3"
_DEFAULT_ARCHITECTURES = ["flat", "naive_rag"]
_DEFAULT_DATASETS = ["qasper", "novelqa"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _linearise_paper(paper: dict[str, Any]) -> str:
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


# ──────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────

def load_qasper_calibration(data_root: Path) -> list[dict[str, Any]]:
    """Return one work item per calibration question with full context.

    Each item carries:
      - dataset, paper_id, question_id
      - question text + (no options for QASPER)
      - full document text (linearised)
      - gold_answers: list of free-form reference answers
      - gold_evidence_sentences: list of highlighted_evidence sentences
    """
    pool = _load_jsonl(data_root / "qasper" / "calibration_pool.jsonl")
    dev_index = {p["paper_id"]: p for p in _load_jsonl(data_root / "qasper" / "dev.jsonl")}

    items: list[dict[str, Any]] = []
    for entry in pool:
        paper_id = entry["paper_id"]
        question_id = entry["question_id"]
        paper = dev_index.get(paper_id)
        if paper is None:
            continue
        qa = next(
            (q for q in paper.get("qas", []) if q.get("question_id") == question_id),
            None,
        )
        if qa is None:
            continue

        gold_answers: list[str] = []
        gold_evidence: list[str] = []
        for ans in qa.get("answers", []) or []:
            a = ans.get("answer", {})
            if not isinstance(a, dict):
                continue
            # Free-form answer text per QASPER schema: prefer free_form_answer,
            # fall back to extractive_spans joined, then to "Yes"/"No" for yes/no.
            ff = a.get("free_form_answer") or ""
            if ff and ff.strip():
                gold_answers.append(ff.strip())
            elif a.get("extractive_spans"):
                gold_answers.append(" ".join(s for s in a["extractive_spans"] if s))
            elif a.get("yes_no") is True:
                gold_answers.append("Yes")
            elif a.get("yes_no") is False:
                gold_answers.append("No")
            elif a.get("unanswerable"):
                gold_answers.append("")  # empty string == "no answer"

            gold_evidence.extend(s for s in (a.get("highlighted_evidence") or []) if s)

        items.append({
            "dataset": "qasper",
            "paper_id": paper_id,
            "question_id": question_id,
            "question": qa["question"],
            "options": None,
            "document": _linearise_paper(paper),
            "gold_answers": gold_answers,
            "gold_evidence_sentences": gold_evidence,
            "gold_label": None,  # QASPER is free-form; no MC label
        })
    return items


def load_novelqa_calibration(data_root: Path) -> list[dict[str, Any]]:
    """One item per NovelQA calibration question, no gold label.

    NovelQA's public dataset release does not ship gold labels for the
    public-domain questions — answers live behind the leaderboard at
    novelqa.github.io. The dry run logs predictions; scoring is
    deferred to a leaderboard submission. ``gold_label`` is ``None``.
    """
    pool = _load_jsonl(data_root / "novelqa" / "calibration_pool.jsonl")
    full_texts_dir = data_root / "novelqa" / "full_texts"

    items: list[dict[str, Any]] = []
    for q in pool:
        novel_id = q["novel_id"]
        text_path = full_texts_dir / f"{novel_id}.txt"
        if not text_path.exists():
            continue
        document = text_path.read_text(encoding="utf-8")
        items.append({
            "dataset": "novelqa",
            "paper_id": novel_id,
            "question_id": q["question_id"],
            "question": q["Question"],
            "options": q.get("Options") or {},
            "document": document,
            "gold_answers": [],            # not available locally
            "gold_evidence_sentences": [], # not available
            "gold_label": None,            # leaderboard-only
        })
    return items


# ──────────────────────────────────────────────────────────────────────
# Per-item invocation
# ──────────────────────────────────────────────────────────────────────

def _invoke_architecture(
    architecture: str,
    item: dict[str, Any],
    *,
    answerer,
    answerer_model: str,
    embedder: OllamaEmbedder | None,
    chunker: SentenceBoundaryChunker | None,
    ledger: CostLedger,
    naive_rag_top_k: int,
) -> ArchitectureResult:
    if architecture == "flat":
        return run_flat(
            document=item["document"],
            query=item["question"],
            options=item["options"],
            answerer=answerer,
            answerer_model=answerer_model,
            ledger=ledger,
            cache_control=CacheControl.EPHEMERAL_5MIN,
        )
    if architecture == "naive_rag":
        if embedder is None or chunker is None:
            raise RuntimeError("naive_rag requires embedder + chunker")
        return run_naive_rag(
            document=item["document"],
            query=item["question"],
            options=item["options"],
            answerer=answerer,
            answerer_model=answerer_model,
            embedder=embedder,
            chunker=chunker,
            ledger=ledger,
            top_k=naive_rag_top_k,
            cache_control=CacheControl.EPHEMERAL_5MIN,
        )
    raise ValueError(f"unsupported architecture: {architecture}")


def _score_item(
    item: dict[str, Any], result: ArchitectureResult
) -> dict[str, float | None]:
    """Compute the metrics that have local labels for this item.

    QASPER → answer_f1 (multi-reference) + evidence_f1.
    NovelQA → no local labels; predicted_letter recorded but no metric.
    """
    if item["dataset"] == "qasper":
        af1 = answer_f1_against_references(
            result.predicted_answer, item["gold_answers"]
        )
        ef1 = evidence_f1(
            result.retrieved_evidence_sentences,
            item["gold_evidence_sentences"],
        )
        return {"answer_f1": af1, "evidence_f1": ef1, "accuracy": None}
    if item["dataset"] == "novelqa":
        # We can still parse + record the chosen letter even though
        # we can't score it locally.
        opts = item.get("options") or {}
        opt_text_list = [opts[k] for k in sorted(opts.keys())]
        predicted_letter = parse_mc_answer(result.predicted_answer, opt_text_list)
        return {
            "answer_f1": None,
            "evidence_f1": None,
            "accuracy": None,
            "predicted_letter": predicted_letter,
        }
    return {"answer_f1": None, "evidence_f1": None, "accuracy": None}


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def run_dry_run(
    *,
    architectures: list[str],
    datasets: list[str],
    answerer_provider: str,
    answerer_model: str,
    embedder_model: str,
    naive_rag_top_k: int,
    data_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    if "qasper" in datasets:
        items.extend(load_qasper_calibration(data_root))
    if "novelqa" in datasets:
        items.extend(load_novelqa_calibration(data_root))

    if not items:
        raise SystemExit("calibration pool is empty; run make build-calibration")

    answerer = get_provider(answerer_provider)
    embedder = OllamaEmbedder(model=embedder_model) if "naive_rag" in architectures else None
    chunker = SentenceBoundaryChunker(chunk_size_tokens=384, overlap_tokens=0) if "naive_rag" in architectures else None

    run_id = new_run_id()
    runs_root = _project_root() / "outputs" / "runs"
    ledger = CostLedger(run_id=run_id, root=runs_root)

    per_arch_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_arch_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failures: list[dict[str, Any]] = []

    print(f"[step3-dry-run] run_id={run_id} items={len(items)} archs={architectures}", file=sys.stderr)

    for item in items:
        for arch in architectures:
            tag = f"{arch}/{item['dataset']}/{item['paper_id']}/{item['question_id']}"
            try:
                result = _invoke_architecture(
                    arch,
                    item,
                    answerer=answerer,
                    answerer_model=answerer_model,
                    embedder=embedder,
                    chunker=chunker,
                    ledger=ledger,
                    naive_rag_top_k=naive_rag_top_k,
                )
            except Exception as exc:
                failures.append({
                    "architecture": arch,
                    "dataset": item["dataset"],
                    "paper_id": item["paper_id"],
                    "question_id": item["question_id"],
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"[step3-dry-run] FAIL {tag}: {exc!r}", file=sys.stderr)
                continue

            scores = _score_item(item, result)
            row = {
                "dataset": item["dataset"],
                "paper_id": item["paper_id"],
                "question_id": item["question_id"],
                "question": item["question"],
                "predicted_answer": result.predicted_answer,
                "retrieved_chunks_count": len(result.retrieved_evidence_sentences),
                **scores,
            }
            per_arch_predictions[arch].append(row)
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    per_arch_scores[arch][metric].append(float(value))
            print(
                f"[step3-dry-run] OK   {tag}  "
                + " ".join(
                    f"{m}={v:.3f}" for m, v in scores.items()
                    if isinstance(v, (int, float))
                ),
                file=sys.stderr,
            )

    # Write per-arch predictions to ledger run dir.
    for arch, rows in per_arch_predictions.items():
        path = ledger.run_dir / f"{arch}_predictions.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False))
                fh.write("\n")

    # Macro averages.
    macro: dict[str, dict[str, float]] = {}
    for arch, scores in per_arch_scores.items():
        macro[arch] = {
            metric: round(sum(vals) / len(vals), 4) if vals else 0.0
            for metric, vals in scores.items()
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_3_dry_run_{timestamp}.json"
    summary = {
        "timestamp_utc": timestamp,
        "run_id": run_id,
        "architectures": architectures,
        "datasets": datasets,
        "answerer_provider": answerer_provider,
        "answerer_model": answerer_model,
        "embedder_model": embedder_model,
        "naive_rag_top_k": naive_rag_top_k,
        "items_count": len(items),
        "per_arch_counts": {
            arch: len(rows) for arch, rows in per_arch_predictions.items()
        },
        "per_arch_macro": macro,
        "failures_count": len(failures),
        "failures": failures,
        "ledger_path": str(ledger.path),
        "predictions_dir": str(ledger.run_dir),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["verdict_path"] = str(out_path)
    return summary


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=_DEFAULT_ARCHITECTURES,
        choices=["flat", "naive_rag", "raptor", "graphrag"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=_DEFAULT_DATASETS,
        choices=["qasper", "novelqa"],
    )
    parser.add_argument("--answerer-provider", default=_DEFAULT_ANSWERER_PROVIDER)
    parser.add_argument("--answerer-model", default=_DEFAULT_ANSWERER_MODEL)
    parser.add_argument("--embedder-model", default=_DEFAULT_EMBEDDER_MODEL)
    parser.add_argument("--naive-rag-top-k", type=int, default=8)
    parser.add_argument("--data-root", type=Path, default=_project_root() / "data")
    parser.add_argument(
        "--out", type=Path, default=_project_root() / "outputs" / "sanity"
    )
    args = parser.parse_args()

    summary = run_dry_run(
        architectures=args.architectures,
        datasets=args.datasets,
        answerer_provider=args.answerer_provider,
        answerer_model=args.answerer_model,
        embedder_model=args.embedder_model,
        naive_rag_top_k=args.naive_rag_top_k,
        data_root=args.data_root,
        out_dir=args.out,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nWrote verdict: {summary['verdict_path']}", file=sys.stderr)
    return 0 if summary["failures_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
