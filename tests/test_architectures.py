"""Tests for pilot.architectures runners with mocked providers.

These tests verify the cost-accounting contract — every architecture
writes one or more uniformly-shaped CostLedger rows per
(architecture, query) — without touching any live API. The mocks
return canned ProviderResults; assertions cover:
  - the right number of ledger rows are written per run,
  - each row carries non-empty prompt_hash + provider_request_id,
  - the right architecture and stage strings appear,
  - the predicted answer + retrieved evidence are surfaced correctly.

The retrieval / clustering / ranking logic itself is not mocked —
chunkers and cosine top-k run against the real implementations
because they are pure deterministic Python.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.architectures import run_flat, run_naive_rag
from pilot.encoders import OllamaEmbedder, SentenceBoundaryChunker
from pilot.encoders.ollama import EmbeddingResult
from pilot.ledger import CostLedger
from pilot.providers.base import AnswererProvider, CacheControl, ProviderResult


# ──────────────────────────────────────────────────────────────────────
# Mocks
# ──────────────────────────────────────────────────────────────────────

class _StubAnswerer(AnswererProvider):
    """Records every call and returns a canned ProviderResult."""

    name = "stub"

    def __init__(self, response_text: str = "MOCK_ANSWER"):
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        cache_control: CacheControl = CacheControl.DISABLED,
    ) -> ProviderResult:
        self.calls.append({
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "cache_control": cache_control,
        })
        return ProviderResult(
            text=self.response_text,
            uncached_input_tokens=len(prompt) // 4,
            cached_input_tokens=0,
            output_tokens=len(self.response_text) // 4,
            provider_request_id=f"req_stub_{len(self.calls)}",
            wallclock_s=0.001,
        )


class _StubEmbedder(OllamaEmbedder):
    """Returns deterministic length-4 vectors based on the input hash;
    bypasses the HTTP layer entirely so tests don't touch Ollama."""

    def __init__(self, model: str = "stub-embedder"):
        self.model = model
        self.base_url = "stub://"
        self.timeout_s = 0
        self.batch_size = 32
        self._client = MagicMock()
        self.calls: list[list[str]] = []

    def embed(self, texts):
        items = list(texts)
        self.calls.append(items)
        # Pseudo-deterministic small vectors so cosine ranking is well-defined.
        embeddings = []
        for i, text in enumerate(items):
            v = [(hash(text) % 100) / 100.0, len(text) % 7 / 7.0, i / 10.0, 0.5]
            embeddings.append(v)
        return EmbeddingResult(model=self.model, embeddings=embeddings)

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────────────
# run_flat
# ──────────────────────────────────────────────────────────────────────

class TestRunFlat:
    def test_writes_exactly_one_ledger_row(self, tmp_path: Path):
        answerer = _StubAnswerer(response_text="English")
        ledger = CostLedger(run_id="test-flat-1", root=tmp_path)
        result = run_flat(
            document="The dataset is in English.",
            query="What language is the dataset?",
            options=None,
            answerer=answerer,
            answerer_model="stub-model",
            ledger=ledger,
        )
        rows = ledger.read()
        assert len(rows) == 1
        assert rows[0].architecture == "flat"
        assert rows[0].stage == "generate"
        assert rows[0].model == "stub-model"
        assert rows[0].prompt_hash, "prompt_hash should be non-empty"
        assert rows[0].provider_request_id == "req_stub_1"
        assert result.predicted_answer == "English"
        assert result.architecture == "flat"
        assert result.retrieved_evidence_sentences == []

    def test_renders_freeform_template_when_no_options(self, tmp_path: Path):
        answerer = _StubAnswerer()
        ledger = CostLedger(run_id="test-flat-2", root=tmp_path)
        run_flat(
            document="The capital of France is Paris.",
            query="What is the capital of France?",
            options=None,
            answerer=answerer,
            answerer_model="m",
            ledger=ledger,
        )
        prompt = answerer.calls[0]["prompt"]
        assert "Options:" not in prompt
        assert "What is the capital of France?" in prompt

    def test_renders_mc_template_when_options_present(self, tmp_path: Path):
        answerer = _StubAnswerer(response_text="A")
        ledger = CostLedger(run_id="test-flat-3", root=tmp_path)
        run_flat(
            document="Document text.",
            query="Pick one.",
            options={"A": "Option Alpha", "B": "Option Beta"},
            answerer=answerer,
            answerer_model="m",
            ledger=ledger,
        )
        prompt = answerer.calls[0]["prompt"]
        assert "Options:" in prompt
        assert "A. Option Alpha" in prompt
        assert "B. Option Beta" in prompt

    def test_temperature_pinned_to_zero(self, tmp_path: Path):
        answerer = _StubAnswerer()
        ledger = CostLedger(run_id="test-flat-4", root=tmp_path)
        run_flat(
            document="x", query="y", options=None,
            answerer=answerer, answerer_model="m", ledger=ledger,
        )
        assert answerer.calls[0]["temperature"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# run_naive_rag
# ──────────────────────────────────────────────────────────────────────

class TestRunNaiveRag:
    def test_writes_three_ledger_rows_preprocess_retrieval_generate(self, tmp_path: Path):
        """First-question invocation writes three rows: the chunk-
        embed build (PREPROCESS, paid once per paper), the query
        embed (RETRIEVAL, paid every question), and the answerer
        call (GENERATE). The build/query split is what makes
        ``C_off^struct`` distinguishable from ``C_on`` in the cost
        model, which the break-even analysis depends on."""
        answerer = _StubAnswerer(response_text="Naive answer")
        embedder = _StubEmbedder()
        chunker = SentenceBoundaryChunker(chunk_size_tokens=64, overlap_tokens=0)
        ledger = CostLedger(run_id="test-rag-1", root=tmp_path)

        # ~10 sentences so chunker produces multiple chunks.
        document = " ".join(
            f"Sentence number {i} talks about topic {i % 3}." for i in range(20)
        )
        result = run_naive_rag(
            document=document,
            query="What is topic 1 about?",
            options=None,
            answerer=answerer,
            answerer_model="stub-model",
            embedder=embedder,
            chunker=chunker,
            ledger=ledger,
            top_k=4,
        )
        rows = ledger.read()
        assert len(rows) == 3
        stages = sorted(r.stage for r in rows)
        assert stages == ["generate", "preprocess", "retrieval"]
        archs = {r.architecture for r in rows}
        assert archs == {"naive_rag"}
        assert result.predicted_answer == "Naive answer"
        # top_k=4 should produce exactly 4 retrieved chunk texts.
        assert len(result.retrieved_evidence_sentences) == 4
        # preprocessing_state must be populated so the dispatcher can
        # cache the chunk-embed index for subsequent questions.
        assert result.preprocessing_state is not None

    def test_cache_hit_skips_chunk_embed_writes_two_rows(self, tmp_path: Path):
        """On a cache hit, only retrieval (query embed) + generate
        land in the ledger — the chunk-embed PREPROCESS row was
        already paid on the first question for this paper."""
        answerer = _StubAnswerer(response_text="cached answer")
        embedder = _StubEmbedder()
        chunker = SentenceBoundaryChunker(chunk_size_tokens=64, overlap_tokens=0)
        ledger = CostLedger(run_id="test-rag-cache", root=tmp_path)

        document = " ".join(
            f"Sentence number {i} talks about topic {i % 3}." for i in range(20)
        )
        first = run_naive_rag(
            document=document, query="q1", options=None,
            answerer=answerer, answerer_model="m",
            embedder=embedder, chunker=chunker, ledger=ledger,
            top_k=4,
        )
        assert first.preprocessing_state is not None
        rows_after_first = len(ledger.read())

        second = run_naive_rag(
            document=document, query="q2", options=None,
            answerer=answerer, answerer_model="m",
            embedder=embedder, chunker=chunker, ledger=ledger,
            top_k=4,
            cached_state=first.preprocessing_state,
        )
        rows_after_second = ledger.read()
        new_rows = rows_after_second[rows_after_first:]
        assert len(new_rows) == 2
        new_stages = sorted(r.stage for r in new_rows)
        assert new_stages == ["generate", "retrieval"]
        # The second invocation must have reused the cached state, not
        # produced a fresh one. preprocessing_state identity is the
        # contract the dispatcher relies on.
        assert second.preprocessing_state is first.preprocessing_state

    def test_top_k_caps_retrieved_evidence(self, tmp_path: Path):
        answerer = _StubAnswerer()
        embedder = _StubEmbedder()
        chunker = SentenceBoundaryChunker(chunk_size_tokens=32, overlap_tokens=0)
        ledger = CostLedger(run_id="test-rag-2", root=tmp_path)

        # Long document → many chunks. top_k=2 should clip to 2.
        document = " ".join(f"Chunk seed sentence number {i}." for i in range(40))
        result = run_naive_rag(
            document=document, query="q", options=None,
            answerer=answerer, answerer_model="m",
            embedder=embedder, chunker=chunker, ledger=ledger,
            top_k=2,
        )
        assert len(result.retrieved_evidence_sentences) == 2

    def test_empty_document_fails_gracefully(self, tmp_path: Path):
        answerer = _StubAnswerer()
        embedder = _StubEmbedder()
        chunker = SentenceBoundaryChunker(chunk_size_tokens=64, overlap_tokens=0)
        ledger = CostLedger(run_id="test-rag-3", root=tmp_path)

        result = run_naive_rag(
            document="", query="q", options=None,
            answerer=answerer, answerer_model="m",
            embedder=embedder, chunker=chunker, ledger=ledger,
        )
        assert result.failed is True
        assert result.failure_reason == "document_produced_no_chunks"
        # No ledger rows written when the architecture short-circuits.
        assert ledger.read() == []
        # Answerer never called.
        assert answerer.calls == []

    def test_query_embedded_in_separate_retrieval_call(self, tmp_path: Path):
        """The query embedding must land in its own embed call (one
        per question, ``Stage.RETRIEVAL``), not bundled with the
        chunk-embed build (``Stage.PREPROCESS``). Without this split
        the per-question retrieval cost is overcounted by the
        chunk-embed contribution.
        """
        answerer = _StubAnswerer()
        embedder = _StubEmbedder()
        chunker = SentenceBoundaryChunker(chunk_size_tokens=64, overlap_tokens=0)
        ledger = CostLedger(run_id="test-rag-4", root=tmp_path)

        run_naive_rag(
            document="Chunk one. Chunk two. Chunk three.",
            query="UNIQUE_QUERY_TOKEN",
            options=None,
            answerer=answerer, answerer_model="m",
            embedder=embedder, chunker=chunker, ledger=ledger,
            top_k=2,
        )
        # The embedder receives two separate batches:
        #   batch 0 = chunk texts (no query)
        #   batch 1 = [query]
        assert len(embedder.calls) == 2
        chunk_batch, query_batch = embedder.calls
        assert "UNIQUE_QUERY_TOKEN" not in chunk_batch
        assert query_batch == ["UNIQUE_QUERY_TOKEN"]

    def test_ledger_rows_share_prompt_hash_with_call(self, tmp_path: Path):
        """The prompt_hash on the GENERATE row must equal sha256 of
        the actual prompt sent to the answerer."""
        from pilot.ledger import sha256_hex

        answerer = _StubAnswerer()
        embedder = _StubEmbedder()
        chunker = SentenceBoundaryChunker(chunk_size_tokens=64, overlap_tokens=0)
        ledger = CostLedger(run_id="test-rag-5", root=tmp_path)

        run_naive_rag(
            document="A. B. C. D. E. F. G. H.",
            query="q?", options=None,
            answerer=answerer, answerer_model="m",
            embedder=embedder, chunker=chunker, ledger=ledger,
            top_k=2,
        )
        rows = ledger.read()
        gen = next(r for r in rows if r.stage == "generate")
        sent_prompt = answerer.calls[0]["prompt"]
        assert gen.prompt_hash == sha256_hex(sent_prompt)
