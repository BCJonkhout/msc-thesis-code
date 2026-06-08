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
import hashlib
import json
import os
import pickle
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pilot.architectures import ArchitectureResult, run_flat, run_naive_rag
from pilot.architectures.graphrag import run_graphrag
from pilot.architectures.raptor import CLUSTERING_SEED, run_raptor
from pilot.encoders import OllamaEmbedder, SentenceBoundaryChunker
from pilot.env import load_env
from pilot.eval import (
    answer_f1_against_references,
    accuracy,
    evidence_f1,
    parse_mc_answer,
)
from pilot.ledger import CostLedger, new_run_id
from pilot.preprocess_cache import (
    CacheRequiredMiss,
    build_cache_key_inputs,
    capture_build_rows_since,
    default_cache_root,
    hash_cache_key,
    ledger_byte_size,
    load_cache_entry,
    make_build_meta,
    replay_build_ledger,
    save_cache_entry,
)
from pilot.providers import get_provider
from pilot.providers.base import CacheControl


_DEFAULT_ANSWERER_MODEL = "gemini-3.1-pro-preview"
_DEFAULT_ANSWERER_PROVIDER = "gemini"
_DEFAULT_EMBEDDER_MODEL = "bge-m3"
_DEFAULT_ARCHITECTURES = ["flat", "naive_rag"]
_DEFAULT_DATASETS = ["qasper", "novelqa"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _slug(text: str) -> str:
    """Filesystem-safe slug for a model id (strips dots / slashes)."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "x"


def canonical_run_id(
    *,
    answerer_provider: str,
    answerer_model: str,
    summary_provider: str | None,
    summary_model: str | None,
    embedder_model: str,
    datasets: list[str],
    architectures: list[str],
    prompt_style: str,
    naive_rag_top_k: int,
) -> str:
    """Deterministic run id derived from the experiment configuration.

    The same configuration always maps to the same run directory, which
    is what makes crash-resume idempotent: re-invoking the sweep after a
    crash reopens the SAME dir, skips the completed cells, and appends
    only the rest. ``run_index`` is deliberately EXCLUDED so all N
    repeats share one dir (the per-row ``run_index`` field distinguishes
    them and the append-only ledger keeps every repeat's cost rows).
    """
    payload = {
        "answerer_provider": answerer_provider,
        "answerer_model": answerer_model,
        "summary_provider": summary_provider,
        "summary_model": summary_model,
        "embedder_model": embedder_model,
        "datasets": sorted(datasets),
        "architectures": sorted(architectures),
        "prompt_style": prompt_style,
        "naive_rag_top_k": naive_rag_top_k,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"main-{_slug(answerer_model)}-"
        f"{'-'.join(sorted(datasets))}-{prompt_style}-{digest}"
    )


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically write run_manifest.json (tmp + fsync + os.replace).

    The manifest records the full config and a ``status`` field
    (``in_progress`` at startup, ``complete`` at clean end). Launchers
    can read it to tell an in-flight (crashed) run from a finished one
    without depending on the verdict JSON, which only exists on clean
    completion.
    """
    path = run_dir / "run_manifest.json"
    tmp = run_dir / "run_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
    except (OSError, AttributeError):
        pass
    os.replace(tmp, path)


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
    prompt_style: str = "pilot",
    run_index: int = 0,
    summary_answerer=None,
    summary_model: str | None = None,
    cached_state: object | None = None,
) -> ArchitectureResult:
    """Dispatch one (architecture, item) call.

    ``cached_state`` carries the prior call's preprocessing artefact
    for this (architecture, paper_id). For Naive RAG it's the cached
    chunk-embed index; for RAPTOR the built tree; for GraphRAG the
    extracted entity graph + community reports + entity vectors. In
    every case this is the load-bearing optimisation behind the
    repeated-context amortisation in the cost model — build cost is
    paid once per (run, paper, arch), per-question cost stays
    proportional to retrieval + answer only. For ``flat`` the
    parameter is ignored (no preprocessing).
    """
    if architecture == "flat":
        return run_flat(
            document=item["document"],
            query=item["question"],
            options=item["options"],
            answerer=answerer,
            answerer_model=answerer_model,
            ledger=ledger,
            run_index=run_index,
            cache_control=CacheControl.EPHEMERAL_5MIN,
            prompt_style=prompt_style,
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
            run_index=run_index,
            top_k=naive_rag_top_k,
            cache_control=CacheControl.EPHEMERAL_5MIN,
            prompt_style=prompt_style,
            cached_state=cached_state,
        )
    if architecture == "raptor":
        if embedder is None:
            raise RuntimeError("raptor requires embedder")
        return run_raptor(
            document=item["document"],
            query=item["question"],
            options=item["options"],
            answerer=answerer,
            answerer_model=answerer_model,
            # Per pilot plan § 5.8 row #10 the cheap-tier summary is
            # the default for RAPTOR's preprocessing. Default value
            # comes from CLI; ``summary_answerer`` may route the
            # summary call through a different provider when the
            # answerer is non-Google (Phase F extension protocol).
            summary_model=summary_model,
            summary_answerer=summary_answerer,
            embedder=embedder,
            ledger=ledger,
            run_index=run_index,
            cached_state=cached_state,
        )
    if architecture == "graphrag":
        if embedder is None:
            raise RuntimeError("graphrag requires embedder")
        return run_graphrag(
            document=item["document"],
            query=item["question"],
            options=item["options"],
            answerer=answerer,
            answerer_model=answerer_model,
            summary_model=summary_model,
            summary_answerer=summary_answerer,
            embedder=embedder,
            ledger=ledger,
            run_index=run_index,
            cached_state=cached_state,
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

def _load_resume_state(
    run_dir: Path | None,
    architectures: list[str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, list[float]]],
    set[tuple[str, str, str, int]],
]:
    """Load already-completed cells from a run dir's per-arch JSONL files.

    Resume-in-place reads back the SAME run directory's
    ``<arch>_predictions.jsonl`` so a re-invocation skips finished work.
    Returns (per_arch_predictions, per_arch_scores, completed_keys);
    each completed key is the 4-tuple ``(arch, paper_id, question_id,
    run_index)`` so the N repeats are resumed independently. A torn
    trailing line (partial write at a crash) is skipped with a warning;
    interior corruption re-raises.
    """
    per_arch_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_arch_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    completed: set[tuple[str, str, str, int]] = set()

    if run_dir is None:
        return per_arch_predictions, per_arch_scores, completed

    for arch in architectures:
        path = run_dir / f"{arch}_predictions.jsonl"
        if not path.exists():
            continue
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Only a torn LAST line is tolerable (the next pass
                # re-executes that one cell); interior corruption is a
                # real problem and must surface.
                if idx == len(lines) - 1:
                    print(
                        f"[step3-dry-run] WARN skipping torn trailing line in "
                        f"{path.name}; it will be re-run",
                        file=sys.stderr,
                    )
                    continue
                raise
            per_arch_predictions[arch].append(row)
            completed.add((
                arch,
                row.get("paper_id", ""),
                row.get("question_id", ""),
                int(row.get("run_index", 0)),
            ))
            for metric, value in row.items():
                if metric in {"answer_f1", "evidence_f1", "accuracy"} and isinstance(value, (int, float)):
                    per_arch_scores[arch][metric].append(float(value))

    return per_arch_predictions, per_arch_scores, completed


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
    prompt_style: str = "pilot",
    run_id: str | None = None,
    run_index: int = 0,
    runs_root: Path | None = None,
    summary_provider: str | None = None,
    summary_model: str | None = None,
    cache_required: bool = False,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    if "qasper" in datasets:
        items.extend(load_qasper_calibration(data_root))
    if "novelqa" in datasets:
        items.extend(load_novelqa_calibration(data_root))

    if not items:
        raise SystemExit("calibration pool is empty; run make build-calibration")

    answerer = get_provider(answerer_provider)

    # Multi-provider routing: when the summary stage runs on a
    # different provider than the answerer (Phase F extension
    # protocol), build a separate summary_answerer. When the
    # summary provider matches the answerer provider, reuse the
    # same provider instance to avoid double-instantiation. When
    # neither summary flag is set, the architecture runners fall
    # back to using the answerer for the summary stage too.
    summary_answerer = None
    if summary_provider is not None:
        summary_answerer = (
            answerer if summary_provider == answerer_provider
            else get_provider(summary_provider)
        )

    needs_embedder = bool({"naive_rag", "raptor", "graphrag"} & set(architectures))
    embedder = OllamaEmbedder(model=embedder_model) if needs_embedder else None
    chunker = SentenceBoundaryChunker(chunk_size_tokens=384, overlap_tokens=0) if "naive_rag" in architectures else None

    runs_root = runs_root if runs_root is not None else _project_root() / "outputs" / "runs"
    if run_id is None:
        run_id = canonical_run_id(
            answerer_provider=answerer_provider,
            answerer_model=answerer_model,
            summary_provider=summary_provider,
            summary_model=summary_model,
            embedder_model=embedder_model,
            datasets=datasets,
            architectures=architectures,
            prompt_style=prompt_style,
            naive_rag_top_k=naive_rag_top_k,
        )
    ledger = CostLedger(run_id=run_id, root=runs_root)
    run_dir = ledger.run_dir

    # Resume-in-place: completed (arch, paper, qid, run_index) cells are
    # read back from THIS run dir's own prediction files. Re-invoking the
    # sweep with the same configuration reopens the same dir, skips the
    # done cells, and appends only the rest — so a laptop crash mid-sweep
    # loses at most the in-flight cell, and the append-only ledger (keyed
    # by run_id) keeps every prior cost row. No copy-forward, no new dir.
    per_arch_predictions, per_arch_scores, completed = _load_resume_state(
        run_dir, architectures
    )
    failures: list[dict[str, Any]] = []

    if completed:
        print(
            f"[step3-dry-run] resume-in-place run_id={run_id} reusing "
            f"{sum(len(v) for v in per_arch_predictions.values())} prior rows "
            f"across {len(per_arch_predictions)} archs",
            file=sys.stderr,
        )

    # Prior rows already live in the per-arch files on disk; open in
    # append mode and write ONLY new cells (each flushed + fsync'd below).
    pred_files: dict[str, Any] = {
        arch: (run_dir / f"{arch}_predictions.jsonl").open("a", encoding="utf-8")
        for arch in architectures
    }

    run_manifest = {
        "run_id": run_id,
        "status": "in_progress",
        "architectures": sorted(architectures),
        "datasets": sorted(datasets),
        "answerer_provider": answerer_provider,
        "answerer_model": answerer_model,
        "summary_provider": summary_provider or answerer_provider,
        "summary_model": summary_model or answerer_model,
        "embedder_model": embedder_model,
        "naive_rag_top_k": naive_rag_top_k,
        "prompt_style": prompt_style,
        "run_index": run_index,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_manifest(run_dir, run_manifest)

    print(
        f"[step3-dry-run] run_id={run_id} run_index={run_index} "
        f"items={len(items)} archs={architectures}",
        file=sys.stderr,
    )

    # Per-(arch, paper_id) preprocessing-state cache. RAPTOR's tree
    # and GraphRAG's knowledge graph are built once per paper and
    # reused across every question on that paper, so the cost ledger
    # records ``C_off^struct`` exactly once per paper instead of once
    # per question. This is the on-disk realisation of the
    # ``C_off^struct / n`` amortisation in the cost model
    # (project.tex § 3.4.1); without it the per-(method, query) cell
    # would over-count preprocessing by a factor of n questions per
    # document, silently inverting the Pareto frontier.
    #
    # Eviction: when we move past the last question on a paper, drop
    # that paper's cache entries to keep the live working set small
    # on long sweeps. The remaining-paper bag is built once up front
    # from the input items.
    preprocessing_cache: dict[tuple[str, str], object] = {}
    remaining_paper_questions: dict[str, int] = defaultdict(int)
    for it in items:
        remaining_paper_questions[it["paper_id"]] += 1

    # Resolved summary model id — needed both for the cache key (so a
    # change in summary model invalidates the cached artefact) and for
    # ``build_meta`` storage on save. Mirrors the inline default the
    # arch runners apply when ``summary_model`` is None.
    resolved_summary_model = summary_model or answerer_model
    disk_cache_root = cache_root if cache_root is not None else default_cache_root()

    try:
        for item in items:
            paper_id = item["paper_id"]
            for arch in architectures:
                tag = f"{arch}/{item['dataset']}/{paper_id}/{item['question_id']}#r{run_index}"
                key = (arch, paper_id, item["question_id"], run_index)
                if key in completed:
                    print(f"[step3-dry-run] SKIP {tag} (already done)", file=sys.stderr)
                    continue
                cache_key = (arch, paper_id)
                cached_state = preprocessing_cache.get(cache_key)
                if cached_state is not None and arch in {"naive_rag", "raptor", "graphrag"}:
                    print(
                        f"[step3-dry-run] HIT  {tag} preprocessing cached (in-process)",
                        file=sys.stderr,
                    )

                # Disk-cache consult for RAPTOR + GraphRAG. Naive RAG's
                # build is fast and deterministic (chunk + BGE-M3
                # embed), so it stays intra-process only. RAPTOR's
                # tree and GraphRAG's graph both depend on a
                # non-deterministic LLM summary/extraction stage, so
                # disk-caching them is what gives cross-candidate
                # determinism on the rerun.
                disk_key_inputs: dict[str, Any] | None = None
                disk_key_hash: str | None = None
                if (
                    cached_state is None
                    and arch in {"raptor", "graphrag"}
                ):
                    disk_key_inputs = build_cache_key_inputs(
                        architecture=arch,
                        paper_id=paper_id,
                        dataset=item["dataset"],
                        summary_model=resolved_summary_model,
                        summary_temperature=0.0,  # pilot lock § 3.4.3
                        encoder_model=embedder_model,
                        # Bind the REAL clustering seed (cluster_utils.
                        # RANDOM_SEED) into the key so a seed change
                        # invalidates cached trees instead of silently
                        # reusing artefacts built under a different seed.
                        seed=CLUSTERING_SEED,
                    )
                    disk_key_hash = hash_cache_key(disk_key_inputs)
                    entry = load_cache_entry(
                        architecture=arch,
                        paper_id=paper_id,
                        key_hash=disk_key_hash,
                        cache_root=disk_cache_root,
                    )
                    if entry is not None:
                        # Disk HIT. Replay the build-cost rows into this
                        # ledger ONLY when the artefact was built by a
                        # DIFFERENT run (cross-candidate reuse), so that
                        # candidate's deployment cost stays realistic.
                        # When build_run_id == this run_id the rows are
                        # already in this ledger (resume-in-place), and
                        # build cost belongs to run_index 0 only — so
                        # replaying on a same-run resume or on run_index>0
                        # would DOUBLE-COUNT. Suppress it in those cases.
                        cached_state = entry.state
                        preprocessing_cache[cache_key] = cached_state
                        same_run = entry.build_meta.get("build_run_id") == run_id
                        if run_index == 0 and not same_run:
                            replayed = replay_build_ledger(
                                ledger=ledger,
                                build_meta=entry.build_meta,
                                target_run_index=run_index,
                            )
                        else:
                            replayed = 0
                        print(
                            f"[step3-dry-run] HIT  {tag} preprocessing cached "
                            f"(disk, source_run={entry.build_meta.get('build_run_id')}, "
                            f"replayed_rows={replayed})",
                            file=sys.stderr,
                        )
                    elif cache_required:
                        # ``--cache-required`` mode: a miss here would
                        # silently rebuild from scratch and re-trigger
                        # the cross-candidate divergence the disk
                        # cache exists to prevent. Abort the whole run
                        # so the operator can pre-populate the cache.
                        raise CacheRequiredMiss(
                            f"cache MISS for {arch}/{paper_id} key={disk_key_hash} "
                            f"under --cache-required; pre-populate the disk cache "
                            f"or drop --cache-required"
                        )
                    else:
                        print(
                            f"[step3-dry-run] MISS {tag} preprocessing cache "
                            f"(disk, key={disk_key_hash}) — building",
                            file=sys.stderr,
                        )

                # Snapshot the ledger position BEFORE the call so we
                # can capture the preprocess rows the runner emits on
                # a build (used to populate build_meta after the call).
                pre_call_ledger_offset = ledger_byte_size(ledger)
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
                        prompt_style=prompt_style,
                        run_index=run_index,
                        summary_answerer=summary_answerer,
                        summary_model=summary_model,
                        cached_state=cached_state,
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

                # Cache the preprocessing artefact for future questions
                # on this paper. Only RAPTOR/GraphRAG return a non-None
                # preprocessing_state; flat / naive_rag have nothing to
                # cache. Stored under (arch, paper_id) so two
                # architectures on the same paper don't collide.
                if result.preprocessing_state is not None:
                    preprocessing_cache[cache_key] = result.preprocessing_state

                # Persist the freshly-built artefact to disk if we
                # missed disk-cache above and the runner produced a
                # state object. Skipping this on intra-process HIT
                # (``disk_key_inputs is None``) avoids overwriting a
                # valid entry with itself; skipping on disk HIT (same
                # condition) avoids re-pickling the already-cached
                # artefact every question.
                if (
                    disk_key_inputs is not None
                    and disk_key_hash is not None
                    and result.preprocessing_state is not None
                    and not result.failed
                ):
                    build_rows = capture_build_rows_since(
                        ledger=ledger,
                        architecture=arch,
                        from_byte_offset=pre_call_ledger_offset,
                    )
                    build_meta = make_build_meta(
                        cache_key_inputs=disk_key_inputs,
                        build_run_id=run_id,
                        summary_model=resolved_summary_model,
                        encoder_model=embedder_model,
                        rows=build_rows,
                    )
                    try:
                        save_cache_entry(
                            architecture=arch,
                            paper_id=paper_id,
                            key_hash=disk_key_hash,
                            state=result.preprocessing_state,
                            build_meta=build_meta,
                            cache_root=disk_cache_root,
                        )
                        print(
                            f"[step3-dry-run] SAVE {tag} preprocessing cache "
                            f"(disk, key={disk_key_hash}, rows={len(build_rows)})",
                            file=sys.stderr,
                        )
                    except (OSError, pickle.PickleError, TypeError) as exc:
                        # Disk save failure is non-fatal — the in-process
                        # cache still works for the remainder of this
                        # run; the next process pays a rebuild. Surface
                        # it so the operator notices.
                        #
                        # ``TypeError`` is included because the
                        # 2026-05-18 Phase G crash surfaced as
                        # ``TypeError: cannot pickle '_thread.lock'
                        # object`` from inside pickle, which does NOT
                        # inherit from ``pickle.PickleError``. After
                        # the __getstate__/__setstate__/rehydrate fix
                        # this should not fire, but catching it keeps
                        # any future drift into an unpicklable field
                        # from killing a multi-question run silently.
                        print(
                            f"[step3-dry-run] WARN disk-cache save failed for {tag}: {exc!r}",
                            file=sys.stderr,
                        )

                scores = _score_item(item, result)
                row = {
                    "dataset": item["dataset"],
                    "paper_id": item["paper_id"],
                    "question_id": item["question_id"],
                    "run_index": run_index,
                    "question": item["question"],
                    "predicted_answer": result.predicted_answer,
                    "retrieved_chunks_count": len(result.retrieved_evidence_sentences),
                    **scores,
                }
                per_arch_predictions[arch].append(row)
                # Crash-safe incremental flush. fsync forces the OS
                # buffer cache to disk so a power loss after the line
                # was written keeps the prediction durable. Cost: a
                # few ms per row vs. minutes of re-running the whole
                # architecture on a hard crash. (Logged once; if the
                # platform doesn't support fsync, swallow the error so
                # the dispatcher keeps moving.)
                pred_files[arch].write(json.dumps(row, ensure_ascii=False) + "\n")
                pred_files[arch].flush()
                try:
                    import os as _os
                    _os.fsync(pred_files[arch].fileno())
                except (OSError, AttributeError):
                    pass
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

            # After all architectures have been invoked for this
            # item, decrement the remaining-question counter for the
            # paper. When it hits zero this paper is fully processed
            # and its cached preprocessing artefacts (potentially
            # several hundred MB for GraphRAG on a 90k-token novel)
            # can be released. With the item-major iteration order the
            # calibration loaders produce, this evicts each paper's
            # cache once and keeps the live working set to ~one paper
            # per architecture.
            remaining_paper_questions[paper_id] -= 1
            if remaining_paper_questions[paper_id] <= 0:
                evicted = [k for k in preprocessing_cache if k[1] == paper_id]
                for k in evicted:
                    del preprocessing_cache[k]
                if evicted:
                    print(
                        f"[step3-dry-run] EVICT preprocessing cache for paper_id={paper_id} "
                        f"({len(evicted)} arch entries)",
                        file=sys.stderr,
                    )
    finally:
        for fh in pred_files.values():
            fh.close()

    # No end-of-loop rewrite: every row was already appended + fsync'd
    # incrementally, and prior rows are durable on disk from earlier
    # passes. A truncate-and-rewrite here would risk destroying the
    # durable file on a crash mid-rewrite (the exact failure resume-in-
    # place exists to avoid).

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
        "run_index": run_index,
        "architectures": architectures,
        "datasets": datasets,
        "answerer_provider": answerer_provider,
        "answerer_model": answerer_model,
        "summary_provider": summary_provider or answerer_provider,
        "summary_model": summary_model or answerer_model,
        "embedder_model": embedder_model,
        "naive_rag_top_k": naive_rag_top_k,
        "prompt_style": prompt_style,
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

    # Flip the manifest to complete so a launcher can tell a finished
    # run_index from an in-flight (crashed) one without the verdict JSON.
    run_manifest["status"] = "complete"
    run_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_manifest["failures_count"] = len(failures)
    _write_manifest(run_dir, run_manifest)
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
    parser.add_argument(
        "--summary-provider",
        default=None,
        help=(
            "Provider for the summary stage in RAPTOR + GraphRAG "
            "(entity extraction, per-community summary, recursive "
            "summary). Defaults to --answerer-provider; override "
            "to route the summary stage through a different "
            "provider while keeping the answerer fixed (Phase F "
            "extension protocol — e.g. Grok answerer + Gemini "
            "Flash Lite summary). Has no effect on Flat / "
            "Naive RAG (no summary stage)."
        ),
    )
    parser.add_argument(
        "--summary-model",
        default=None,
        help=(
            "Model id for the summary stage. Defaults to "
            "--answerer-model. Pair with --summary-provider when "
            "the summary model lives on a different provider "
            "than the answerer."
        ),
    )
    parser.add_argument("--embedder-model", default=_DEFAULT_EMBEDDER_MODEL)
    parser.add_argument("--naive-rag-top-k", type=int, default=8)
    parser.add_argument(
        "--prompt-style",
        choices=["pilot", "literature"],
        default="pilot",
        help=(
            "Free-form QA prompt style. 'pilot' (default) uses the "
            "abstention-encouraging template; 'literature' uses the "
            "concise-answer template comparable to RAPTOR / Self-RAG / "
            "RAG published QASPER baselines. MC questions are unaffected "
            "in either mode."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=_project_root() / "data")
    parser.add_argument(
        "--out", type=Path, default=_project_root() / "outputs" / "sanity"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Explicit run directory name under outputs/runs/. Defaults "
            "to a deterministic id derived from the configuration "
            "(answerer, datasets, architectures, prompt_style, ...), so "
            "re-invoking the same sweep resumes IN PLACE: completed "
            "(arch, paper, question, run_index) cells are skipped and "
            "only the missing ones run. The append-only ledger keeps "
            "every prior cost row."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help=(
            "Number of repeats (run_index 0..N-1) to execute into the "
            "same run dir, for across-run variance. The main study uses "
            "N=5. Ignored when --run-index is given."
        ),
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help=(
            "Run a SINGLE repeat at this run_index (for launching the N "
            "repeats as separate parallel processes). Overrides "
            "--num-runs. Cost accounting counts run_index 0 only."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Compatibility shim: resume IN PLACE into this exact run "
            "directory (outputs/runs/<run_id>/) rather than deriving the "
            "dir from config. Equivalent to --run-id <dirname> with the "
            "matching runs root. Completed cells are skipped; the ledger "
            "is preserved (append-only). Prefer --run-id / config-derived "
            "resume for new runs."
        ),
    )
    parser.add_argument(
        "--cache-required",
        action="store_true",
        help=(
            "Abort the run on any RAPTOR/GraphRAG disk-cache MISS "
            "instead of silently rebuilding the artefact. Required "
            "for cross-candidate determinism runs: a rebuild would "
            "re-introduce non-determinism in the summary / entity-"
            "extraction stage and invalidate the cross-answerer "
            "comparison the cache exists to enable. Pre-populate "
            "the cache first by running one candidate without this "
            "flag."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help=(
            "Override the on-disk preprocessing cache root. Defaults "
            "to code/outputs/preprocess_cache/. Useful for isolated "
            "test runs or for sharing a cache across multiple "
            "checkout directories."
        ),
    )
    args = parser.parse_args()

    # Resolve the run directory. Resume-in-place: the SAME config maps to
    # the SAME canonical dir, so re-invoking after a crash resumes
    # automatically. --run-id pins an explicit dir; --resume-from <dir>
    # (compat) resumes in place into that exact dir.
    run_id = args.run_id
    runs_root: Path | None = None
    if args.resume_from is not None:
        run_id = args.resume_from.name
        runs_root = args.resume_from.parent

    # Which repeats to run. --run-index pins a single repeat (for parallel
    # launching across processes); otherwise loop 0..num_runs-1 into the
    # shared dir.
    indices = [args.run_index] if args.run_index is not None else list(range(args.num_runs))

    summary: dict[str, Any] = {}
    total_failures = 0
    for ri in indices:
        summary = run_dry_run(
            architectures=args.architectures,
            datasets=args.datasets,
            answerer_provider=args.answerer_provider,
            answerer_model=args.answerer_model,
            embedder_model=args.embedder_model,
            naive_rag_top_k=args.naive_rag_top_k,
            data_root=args.data_root,
            out_dir=args.out,
            prompt_style=args.prompt_style,
            run_id=run_id,
            run_index=ri,
            runs_root=runs_root,
            summary_provider=args.summary_provider,
            summary_model=args.summary_model,
            cache_required=args.cache_required,
            cache_root=args.cache_root,
        )
        # Pin the resolved run_id + root so subsequent repeats share the
        # exact same dir even when it was derived canonically on the first.
        run_id = summary["run_id"]
        if runs_root is None:
            runs_root = Path(summary["predictions_dir"]).parent
        total_failures += summary["failures_count"]
        print(
            f"\nWrote verdict (run_index={ri}): {summary['verdict_path']}",
            file=sys.stderr,
        )

    print(json.dumps(summary, indent=2))
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
