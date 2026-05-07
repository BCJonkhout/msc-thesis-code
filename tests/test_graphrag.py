"""Tests for the from-scratch GraphRAG runner.

Coverage targets:

  - Pipeline-stage helpers (`_parse_extract_json`,
    `_build_graph_and_communities`) are deterministic and behave
    as documented.
  - End-to-end `run_graphrag` with a mocked answerer and embedder
    writes the expected number of ledger rows tagged with the
    right (architecture, stage) pairs:
       * one PREPROCESS per chunk (entity extraction)
       * one PREPROCESS per community (community summary)
       * one RETRIEVAL (the local-search query embed)
       * one GENERATE (the final answer call)
  - Graceful failure on empty / no-entity inputs (no ledger
    rows from skipped stages, ArchitectureResult.failed=True).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.architectures.graphrag import (
    _Entity,
    _Relationship,
    _build_graph_and_communities,
    _parse_extract_json,
    run_graphrag,
)
from pilot.encoders import OllamaEmbedder
from pilot.encoders.ollama import EmbeddingResult
from pilot.ledger import CostLedger
from pilot.providers.base import AnswererProvider, CacheControl, ProviderResult


# ──────────────────────────────────────────────────────────────────────
# Mocks (mirroring tests/test_architectures.py)
# ──────────────────────────────────────────────────────────────────────

class _ScriptedAnswerer(AnswererProvider):
    """Returns successive canned responses from a list. Each call
    advances one position; once exhausted, returns the last item.
    Useful for staging entity-extraction JSON on early calls and
    community summaries / final answers on later calls.
    """

    name = "scripted"

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self._idx = 0

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
        text = self.responses[min(self._idx, len(self.responses) - 1)]
        self._idx += 1
        self.calls.append({"prompt": prompt, "model": model, "response": text})
        return ProviderResult(
            text=text,
            uncached_input_tokens=len(prompt) // 4,
            cached_input_tokens=0,
            output_tokens=len(text) // 4,
            provider_request_id=f"req_scripted_{self._idx}",
            wallclock_s=0.001,
        )


class _StubEmbedder(OllamaEmbedder):
    """Deterministic length-4 vectors via input hash. Bypasses HTTP."""

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
        embeddings = []
        for i, text in enumerate(items):
            v = [(hash(text) % 100) / 100.0, len(text) % 7 / 7.0, i / 10.0, 0.5]
            embeddings.append(v)
        return EmbeddingResult(model=self.model, embeddings=embeddings)

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────────────
# _parse_extract_json
# ──────────────────────────────────────────────────────────────────────

class TestParseExtractJson:
    def test_strict_json(self):
        raw = '{"entities": [{"name": "Alice", "type": "person", "description": "x"}], "relationships": []}'
        out = _parse_extract_json(raw)
        assert len(out["entities"]) == 1
        assert out["entities"][0]["name"] == "Alice"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"entities": [{"name": "Bob"}], "relationships": []}\n```'
        out = _parse_extract_json(raw)
        assert out["entities"][0]["name"] == "Bob"

    def test_prose_around_json(self):
        raw = (
            "Sure, here's the extraction:\n"
            '{"entities": [{"name": "Carol"}], "relationships": []}\n'
            "Hope that helps!"
        )
        out = _parse_extract_json(raw)
        assert out["entities"][0]["name"] == "Carol"

    def test_empty_input_returns_empty_lists(self):
        assert _parse_extract_json("") == {"entities": [], "relationships": []}
        assert _parse_extract_json("   \n  ") == {"entities": [], "relationships": []}

    def test_unparseable_returns_empty_lists(self):
        assert _parse_extract_json("not json at all") == {
            "entities": [], "relationships": []
        }


# ──────────────────────────────────────────────────────────────────────
# _build_graph_and_communities
# ──────────────────────────────────────────────────────────────────────

class TestBuildGraphAndCommunities:
    def test_empty_inputs_return_empty_graph(self):
        g, communities = _build_graph_and_communities([], [])
        assert g.number_of_nodes() == 0
        assert communities == []

    def test_nodes_carry_type_and_description(self):
        ents = [
            _Entity(name="Alice", type="person", description="A protagonist."),
            _Entity(name="London", type="place", description="A city."),
        ]
        rels = [_Relationship(source="Alice", target="London", description="lives in")]
        g, _ = _build_graph_and_communities(ents, rels)
        assert g.number_of_nodes() == 2
        assert g.nodes["Alice"]["type"] == "person"
        assert g.nodes["Alice"]["description"] == "A protagonist."
        assert g.nodes["London"]["type"] == "place"

    def test_edge_weights_count_co_occurrences(self):
        # The paper defines edge weight as the number of duplicate
        # relationship instances detected. Three (Alice, London)
        # triples should produce one edge with weight=3.
        ents = [_Entity(name=n, type="x", description="") for n in ("Alice", "London")]
        rels = [
            _Relationship(source="Alice", target="London", description="r1"),
            _Relationship(source="Alice", target="London", description="r2"),
            _Relationship(source="London", target="Alice", description="r3"),  # reverse order normalises
        ]
        g, _ = _build_graph_and_communities(ents, rels)
        assert g.number_of_edges() == 1
        edge = g.edges["Alice", "London"]
        assert edge["weight"] == 3
        assert "r1" in edge["description"] and "r2" in edge["description"] and "r3" in edge["description"]

    def test_louvain_seed_is_deterministic(self):
        # Same inputs + same seed → identical communities across runs.
        ents = [_Entity(name=str(i), type="x", description="") for i in range(20)]
        rels = []
        # Add a clique on each half to force two communities.
        for i in range(10):
            for j in range(i + 1, 10):
                rels.append(_Relationship(source=str(i), target=str(j), description=""))
        for i in range(10, 20):
            for j in range(i + 1, 20):
                rels.append(_Relationship(source=str(i), target=str(j), description=""))
        a_g, a_communities = _build_graph_and_communities(ents, rels, seed=42)
        b_g, b_communities = _build_graph_and_communities(ents, rels, seed=42)
        assert [sorted(c) for c in a_communities] == [sorted(c) for c in b_communities]


# ──────────────────────────────────────────────────────────────────────
# run_graphrag end-to-end with mocked providers
# ──────────────────────────────────────────────────────────────────────

class TestRunGraphragEndToEnd:
    def test_one_chunk_one_entity_writes_expected_ledger_rows(self, tmp_path: Path):
        """Smallest sensible input: a tiny document → 1 chunk → 1
        entity-extraction call → 1 community → 1 community-summary
        call → 1 query embed → 1 final answer call.
        """
        # Document short enough to fit in one 600-token chunk. ~20 words.
        document = "Alice met Bob in London. They had a long conversation about philosophy and art."
        # Scripted answers: extraction, community summary, final answer.
        answerer = _ScriptedAnswerer([
            json.dumps({
                "entities": [
                    {"name": "Alice", "type": "person", "description": "First protagonist."},
                    {"name": "Bob", "type": "person", "description": "Second character."},
                    {"name": "London", "type": "place", "description": "City."},
                ],
                "relationships": [
                    {"source": "Alice", "target": "Bob", "description": "met"},
                    {"source": "Alice", "target": "London", "description": "in"},
                ],
            }),
            "A community of two people meeting in a city.",  # community summary
            "Alice and Bob met in London.",  # final answer
        ])
        embedder = _StubEmbedder()
        ledger = CostLedger(run_id="test-graphrag-1", root=tmp_path)

        result = run_graphrag(
            document=document,
            query="Where did Alice and Bob meet?",
            options=None,
            answerer=answerer,
            answerer_model="stub-model",
            embedder=embedder,
            ledger=ledger,
        )

        assert result.failed is False
        assert result.architecture == "graphrag"
        assert result.predicted_answer == "Alice and Bob met in London."

        rows = ledger.read()
        # One PREPROCESS per chunk (1) + one per community (>=1) + one RETRIEVAL + one GENERATE.
        stages = [r.stage for r in rows]
        assert stages.count("retrieval") == 1
        assert stages.count("generate") == 1
        # At least one PREPROCESS for entity extraction. Community summary
        # depends on Louvain partition; for 3 nodes + 2 edges it's typically 1.
        assert stages.count("preprocess") >= 2
        assert all(r.architecture == "graphrag" for r in rows)
        # The final-answer prompt is what the answerer last received.
        final_prompt = answerer.calls[-1]["prompt"]
        assert "Alice and Bob meet" in final_prompt or "Where did Alice" in final_prompt

    def test_empty_document_fails_gracefully(self, tmp_path: Path):
        answerer = _ScriptedAnswerer(["{}"])
        embedder = _StubEmbedder()
        ledger = CostLedger(run_id="test-graphrag-2", root=tmp_path)
        result = run_graphrag(
            document="", query="?", options=None,
            answerer=answerer, answerer_model="m",
            embedder=embedder, ledger=ledger,
        )
        assert result.failed is True
        assert result.failure_reason == "document_produced_no_chunks"
        # No ledger rows; no answerer calls.
        assert ledger.read() == []
        assert answerer.calls == []

    def test_no_entities_extracted_fails_gracefully(self, tmp_path: Path):
        # A document with one chunk but the LLM returns no entities.
        document = "x " * 200  # forces at least one chunk
        answerer = _ScriptedAnswerer([
            json.dumps({"entities": [], "relationships": []}),
        ])
        embedder = _StubEmbedder()
        ledger = CostLedger(run_id="test-graphrag-3", root=tmp_path)
        result = run_graphrag(
            document=document, query="?", options=None,
            answerer=answerer, answerer_model="m",
            embedder=embedder, ledger=ledger,
        )
        assert result.failed is True
        assert result.failure_reason == "no_entities_extracted"
        # The extraction call was made but no community / final / embed call.
        rows = ledger.read()
        assert len(rows) == 1
        assert rows[0].stage == "preprocess"

    def test_returns_retrieved_summaries_as_evidence(self, tmp_path: Path):
        document = "Alice met Bob in London. They went to the British Museum."
        answerer = _ScriptedAnswerer([
            json.dumps({
                "entities": [
                    {"name": "Alice", "type": "person", "description": "x"},
                    {"name": "Bob", "type": "person", "description": "y"},
                    {"name": "London", "type": "place", "description": "z"},
                    {"name": "British Museum", "type": "place", "description": "w"},
                ],
                "relationships": [
                    {"source": "Alice", "target": "Bob", "description": "met"},
                    {"source": "Alice", "target": "London", "description": "in"},
                    {"source": "British Museum", "target": "London", "description": "in"},
                ],
            }),
            "First community report content.",
            "Second community report content.",
            "Final answer.",
        ])
        embedder = _StubEmbedder()
        ledger = CostLedger(run_id="test-graphrag-4", root=tmp_path)
        result = run_graphrag(
            document=document, query="q?", options=None,
            answerer=answerer, answerer_model="m",
            embedder=embedder, ledger=ledger,
        )
        assert result.failed is False
        # retrieved_evidence_sentences should carry one or more
        # community-report summaries that the local search picked.
        assert len(result.retrieved_evidence_sentences) >= 1
        for s in result.retrieved_evidence_sentences:
            assert "community report" in s.lower()
