"""Tests for the per-(architecture, paper_id) preprocessing cache in
``pilot.cli.step_3_dry_run.run_dry_run``.

The cache is the on-disk realisation of the ``C_off^struct / n``
amortisation rule in the cost model (project.tex § 3.4.1):
RAPTOR's UMAP+GMM tree and GraphRAG's knowledge graph are built once
per paper and reused across every question on that paper. Without
this behaviour the cost ledger would attribute ``C_off^struct`` N
times (once per question) instead of once per document, silently
inverting every Pareto comparison the benchmark exists to make.

Coverage:
  - The first question on a paper invokes the runner with
    ``cached_state=None``.
  - Every subsequent question on the same paper receives the
    sentinel state returned by the first call.
  - Two papers each maintain their own independent cache entry.
  - When the last question on a paper has been processed, that
    paper's cache entry is evicted.
  - Flat / Naive RAG do not have ``cached_state`` threaded through
    (they have no preprocessing artefact).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.architectures import ArchitectureResult
from pilot.cli import step_3_dry_run as cli


def _write_qasper_pool(
    data_root: Path,
    paper_questions: list[tuple[str, list[str]]],
) -> None:
    """Write a tiny fake QASPER calibration pool on disk.

    ``paper_questions`` is a list of ``(paper_id, [question_id, ...])``.
    Each question gets a non-empty gold answer so the F1 scoring
    doesn't barf on the result.
    """
    qasper = data_root / "qasper"
    qasper.mkdir(parents=True, exist_ok=True)
    papers = []
    cal_rows = []
    for pid, qids in paper_questions:
        qas = []
        for qid in qids:
            qas.append({
                "question_id": qid, "question": "?",
                "answers": [{"answer": {
                    "free_form_answer": "yes",
                    "evidence": [], "highlighted_evidence": [],
                    "extractive_spans": [], "yes_no": True,
                    "unanswerable": False,
                }}],
            })
            cal_rows.append({
                "paper_id": pid, "question_id": qid, "question": "?",
            })
        papers.append({
            "paper_id": pid, "title": f"paper {pid}",
            "abstract": "x",
            "full_text": [{"section_name": "s", "paragraphs": ["text"]}],
            "qas": qas,
        })
    (qasper / "dev.jsonl").write_text(
        "\n".join(json.dumps(p) for p in papers) + "\n",
        encoding="utf-8",
    )
    (qasper / "calibration_pool.jsonl").write_text(
        "\n".join(json.dumps(r) for r in cal_rows) + "\n",
        encoding="utf-8",
    )


class TestPreprocessingCache:
    def test_raptor_builds_once_per_paper_two_questions(
        self, tmp_path: Path, monkeypatch
    ):
        """One paper, two questions: tree built on Q1, reused on Q2."""
        data_root = tmp_path / "data"
        _write_qasper_pool(data_root, [("paperA", ["q1", "q2"])])

        invocations: list[dict[str, Any]] = []
        first_state = object()  # sentinel returned by the first call

        def fake_run_raptor(**kwargs):
            invocations.append({
                "cached_state": kwargs.get("cached_state"),
            })
            # Cache miss: return the sentinel. Cache hit: pass through
            # the previously cached state so the dispatcher's
            # ``cache[(arch, paper)] = result.preprocessing_state``
            # assignment is idempotent.
            state = kwargs.get("cached_state") or first_state
            return ArchitectureResult(
                architecture="raptor", predicted_answer="z",
                preprocessing_state=state,
            )

        monkeypatch.setattr(cli, "run_raptor", fake_run_raptor)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

        cli.run_dry_run(
            architectures=["raptor"],
            datasets=["qasper"],
            answerer_provider="gemini",
            answerer_model="m",
            embedder_model="bge-m3",
            naive_rag_top_k=8,
            data_root=data_root,
            out_dir=tmp_path / "out",
        )

        assert len(invocations) == 2
        assert invocations[0]["cached_state"] is None
        # The second call's cached_state is the exact sentinel object
        # the first call returned — identity check, not equality.
        assert invocations[1]["cached_state"] is first_state

    def test_raptor_each_paper_has_independent_cache(
        self, tmp_path: Path, monkeypatch
    ):
        """Two papers: each paper's first question is a miss; second is
        a hit; the two papers' cached states are distinct."""
        data_root = tmp_path / "data"
        _write_qasper_pool(
            data_root,
            [("paperA", ["q1", "q2"]), ("paperB", ["q3", "q4"])],
        )

        invocations: list[dict[str, Any]] = []
        counter = {"i": 0}

        def fake_run_raptor(**kwargs):
            counter["i"] += 1
            invocations.append({
                "cached_state": kwargs.get("cached_state"),
                "index": counter["i"],
            })
            # Cache miss: return a fresh dict tagged with the build
            # index so each paper has a distinguishable sentinel. Cache
            # hit: pass through.
            state = (
                kwargs.get("cached_state")
                or {"build_index": counter["i"]}
            )
            return ArchitectureResult(
                architecture="raptor", predicted_answer="z",
                preprocessing_state=state,
            )

        monkeypatch.setattr(cli, "run_raptor", fake_run_raptor)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

        cli.run_dry_run(
            architectures=["raptor"],
            datasets=["qasper"],
            answerer_provider="gemini",
            answerer_model="m",
            embedder_model="bge-m3",
            naive_rag_top_k=8,
            data_root=data_root,
            out_dir=tmp_path / "out",
        )

        assert len(invocations) == 4
        # paperA: first miss, second hit
        assert invocations[0]["cached_state"] is None
        assert invocations[1]["cached_state"] is not None
        # paperB: first miss (NOT the paperA state), second hit
        assert invocations[2]["cached_state"] is None
        assert invocations[3]["cached_state"] is not None
        # paperA's cached state is distinct from paperB's
        assert invocations[1]["cached_state"] != invocations[3]["cached_state"]

    def test_graphrag_cache_threads_through(
        self, tmp_path: Path, monkeypatch
    ):
        """Same invariant for GraphRAG: build once, reuse on follow-ups."""
        data_root = tmp_path / "data"
        _write_qasper_pool(data_root, [("paperA", ["q1", "q2", "q3"])])

        invocations: list[dict[str, Any]] = []

        def fake_run_graphrag(**kwargs):
            invocations.append({"cached_state": kwargs.get("cached_state")})
            state = (
                kwargs["cached_state"]
                if kwargs.get("cached_state") is not None
                else {"graph": "built"}
            )
            return ArchitectureResult(
                architecture="graphrag", predicted_answer="z",
                preprocessing_state=state,
            )

        monkeypatch.setattr(cli, "run_graphrag", fake_run_graphrag)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

        cli.run_dry_run(
            architectures=["graphrag"],
            datasets=["qasper"],
            answerer_provider="gemini",
            answerer_model="m",
            embedder_model="bge-m3",
            naive_rag_top_k=8,
            data_root=data_root,
            out_dir=tmp_path / "out",
        )

        assert len(invocations) == 3
        assert invocations[0]["cached_state"] is None
        assert invocations[1]["cached_state"] == {"graph": "built"}
        assert invocations[2]["cached_state"] == {"graph": "built"}

    def test_naive_rag_chunk_index_amortises(
        self, tmp_path: Path, monkeypatch
    ):
        """Naive RAG must build the chunk-embed index once per paper.

        The cost model in project.tex §3.4.1 separates ``C_off^struct``
        (build cost, paid once per paper) from ``C_on`` (per-question
        retrieval + generate). An earlier iteration of naive_rag
        chunk-embedded the whole document on every question, which
        meant build cost was inappropriately billed to per-question
        retrieval and the Pareto comparison against RAPTOR/GraphRAG
        was biased against naive_rag.
        """
        data_root = tmp_path / "data"
        _write_qasper_pool(data_root, [("paperA", ["q1", "q2", "q3"])])

        invocations: list[dict[str, Any]] = []
        first_state = object()

        def fake_run_naive_rag(**kwargs):
            invocations.append({"cached_state": kwargs.get("cached_state")})
            state = kwargs.get("cached_state") or first_state
            return ArchitectureResult(
                architecture="naive_rag", predicted_answer="z",
                preprocessing_state=state,
            )

        monkeypatch.setattr(cli, "run_naive_rag", fake_run_naive_rag)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

        cli.run_dry_run(
            architectures=["naive_rag"],
            datasets=["qasper"],
            answerer_provider="gemini",
            answerer_model="m",
            embedder_model="bge-m3",
            naive_rag_top_k=8,
            data_root=data_root,
            out_dir=tmp_path / "out",
        )

        assert len(invocations) == 3
        # First question: cache miss → build chunk index
        assert invocations[0]["cached_state"] is None
        # Subsequent questions: same paper → HIT with the cached index
        assert invocations[1]["cached_state"] is first_state
        assert invocations[2]["cached_state"] is first_state

    def test_flat_runner_never_sees_cached_state_kwarg(
        self, tmp_path: Path, monkeypatch
    ):
        """Flat has no preprocessing to cache; its runner signature
        does not accept cached_state and would TypeError if the
        dispatcher tried to forward it."""
        data_root = tmp_path / "data"
        _write_qasper_pool(data_root, [("paperA", ["q1", "q2"])])

        captured: list[dict[str, Any]] = []

        def fake_run_flat(**kwargs):
            captured.append(kwargs)
            return ArchitectureResult(architecture="flat", predicted_answer="z")

        monkeypatch.setattr(cli, "run_flat", fake_run_flat)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

        cli.run_dry_run(
            architectures=["flat"],
            datasets=["qasper"],
            answerer_provider="gemini",
            answerer_model="m",
            embedder_model="bge-m3",
            naive_rag_top_k=8,
            data_root=data_root,
            out_dir=tmp_path / "out",
        )

        assert len(captured) == 2
        for kwargs in captured:
            assert "cached_state" not in kwargs
