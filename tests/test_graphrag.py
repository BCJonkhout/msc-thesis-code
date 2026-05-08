"""Tests for the faithful from-scratch GraphRAG local-search runner.

The runner implements Edge et al. 2024 §3.1 + Microsoft graphrag's
`LocalSearchMixedContext.build_context`. Coverage targets:

  - Pipeline-stage helpers (`_parse_extract_json`,
    `_build_graph_and_communities`, `_pack_within_budget`) are
    deterministic and behave as documented.
  - Entity extraction tracks `text_unit_ids` per entity so local
    search can recover chunk provenance.
  - End-to-end `run_graphrag` with mocked providers writes the
    expected ledger-row pattern:
       * (max_gleanings + 1) PREPROCESS rows per chunk for extraction
       * one PREPROCESS per community for the structured report
       * one RETRIEVAL for the local-search query+entity embed
       * one GENERATE for the final answer call
  - Graceful failure on empty / no-entity inputs.
  - Local-search context-builder packs community reports + chunk
    text + entity/relationship lines within the token budget.
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
    _pack_within_budget,
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
        g, communities, edge_tu = _build_graph_and_communities([], [])
        assert g.number_of_nodes() == 0
        assert communities == []
        assert edge_tu == {}

    def test_nodes_carry_type_description_and_text_unit_ids(self):
        ents = [
            _Entity(name="Alice", type="person", description="A protagonist.",
                    text_unit_ids=[0, 2]),
            _Entity(name="London", type="place", description="A city.",
                    text_unit_ids=[2]),
        ]
        rels = [_Relationship(source="Alice", target="London",
                              description="lives in", text_unit_id=2)]
        g, _, edge_tu = _build_graph_and_communities(ents, rels)
        assert g.number_of_nodes() == 2
        assert g.nodes["Alice"]["type"] == "person"
        assert g.nodes["Alice"]["description"] == "A protagonist."
        assert g.nodes["Alice"]["text_unit_ids"] == (0, 2)
        assert g.nodes["London"]["type"] == "place"
        # edge_text_unit_map records the chunk(s) the relationship was
        # extracted from
        assert edge_tu[("Alice", "London")] == [2]

    def test_edge_weights_count_co_occurrences(self):
        # Three (Alice, London) relationships from chunks 0, 1, 2 should
        # produce one undirected edge with weight=3 and edge_text_units
        # tracking [0, 1, 2].
        ents = [_Entity(name=n, type="x", description="") for n in ("Alice", "London")]
        rels = [
            _Relationship(source="Alice", target="London", description="r1", text_unit_id=0),
            _Relationship(source="Alice", target="London", description="r2", text_unit_id=1),
            _Relationship(source="London", target="Alice", description="r3", text_unit_id=2),
        ]
        g, _, edge_tu = _build_graph_and_communities(ents, rels)
        assert g.number_of_edges() == 1
        edge = g.edges["Alice", "London"]
        assert edge["weight"] == 3
        assert "r1" in edge["description"]
        assert "r2" in edge["description"]
        assert "r3" in edge["description"]
        assert sorted(edge_tu[("Alice", "London")]) == [0, 1, 2]

    def test_louvain_seed_is_deterministic(self):
        ents = [_Entity(name=str(i), type="x", description="") for i in range(20)]
        rels = []
        for i in range(10):
            for j in range(i + 1, 10):
                rels.append(_Relationship(source=str(i), target=str(j), description=""))
        for i in range(10, 20):
            for j in range(i + 1, 20):
                rels.append(_Relationship(source=str(i), target=str(j), description=""))
        _ag, a_communities, _ = _build_graph_and_communities(ents, rels, seed=42)
        _bg, b_communities, _ = _build_graph_and_communities(ents, rels, seed=42)
        assert [sorted(c) for c in a_communities] == [sorted(c) for c in b_communities]


# ──────────────────────────────────────────────────────────────────────
# _pack_within_budget — the local-search context-builder helper
# ──────────────────────────────────────────────────────────────────────

class TestPackWithinBudget:
    def test_packs_until_budget_hit(self):
        items = [("a" * 40, 10), ("b" * 80, 20), ("c" * 40, 10)]
        # budget 25 → only the first item (10 tok) + second (20 tok) = 30
        # exceeds; only first fits
        out = _pack_within_budget(items, 25)
        assert out == ["a" * 40]

    def test_takes_full_list_when_budget_exceeds_total(self):
        items = [("x", 5), ("y", 5)]
        out = _pack_within_budget(items, 100)
        assert out == ["x", "y"]

    def test_returns_empty_list_when_first_item_exceeds_budget(self):
        items = [("huge", 1000)]
        out = _pack_within_budget(items, 50)
        assert out == []


# ──────────────────────────────────────────────────────────────────────
# run_graphrag end-to-end with mocked providers
# ──────────────────────────────────────────────────────────────────────

class TestRunGraphragEndToEnd:
    """End-to-end smoke. Each chunk goes through 1 initial extraction
    + 1 gleaning pass (Microsoft default `max_gleanings=1`); the
    gleaning pass that returns empty is recorded but does not re-merge.
    """

    def test_one_chunk_one_entity_writes_expected_ledger_rows(self, tmp_path: Path):
        document = "Alice met Bob in London. They had a long conversation about philosophy and art."
        answerer = _ScriptedAnswerer([
            # Chunk 1 — initial extraction
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
            # Chunk 1 — gleaning pass returns empty (early-exits the loop)
            json.dumps({"entities": [], "relationships": []}),
            # Community report (1 community for this graph)
            "## Title\nMeeting in London\n\n## Summary\nA short community.\n\n## Findings\n- Alice and Bob met in London.",
            # Final answer
            "Alice and Bob met in London.",
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
        stages = [r.stage for r in rows]
        # 1 retrieval (local-search query+entity embed) + 1 generate (final).
        assert stages.count("retrieval") == 1
        assert stages.count("generate") == 1
        # >=2 PREPROCESS for extraction (1 initial + 1 glean) + >=1 community.
        assert stages.count("preprocess") >= 3
        assert all(r.architecture == "graphrag" for r in rows)
        # Final-answer prompt should reference the query.
        final_prompt = answerer.calls[-1]["prompt"]
        assert "Where did Alice" in final_prompt or "Alice and Bob meet" in final_prompt

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
        assert ledger.read() == []
        assert answerer.calls == []

    def test_no_entities_extracted_fails_gracefully(self, tmp_path: Path):
        # One-chunk document. Initial extraction returns empty; gleaning
        # pass also empty → entities dict stays empty → fail.
        document = "x " * 200
        answerer = _ScriptedAnswerer([
            json.dumps({"entities": [], "relationships": []}),
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
        # Both extraction calls are recorded; nothing downstream.
        rows = ledger.read()
        assert len(rows) == 2
        assert all(r.stage == "preprocess" for r in rows)

    def test_evidence_includes_community_reports_or_chunk_text(self, tmp_path: Path):
        """Local search packs community reports + chunk text into the
        context; both should surface as evidence on the result."""
        document = "Alice met Bob in London. They went to the British Museum."
        answerer = _ScriptedAnswerer([
            # Initial extraction (chunk 0)
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
            # Gleaning pass (empty → break)
            json.dumps({"entities": [], "relationships": []}),
            # Community reports (Louvain output is partition-dependent;
            # provide enough scripted responses for up to 2 communities)
            "## Title\nFirst community\n\n## Summary\nFirst.\n\n## Findings\n- One.",
            "## Title\nSecond community\n\n## Summary\nSecond.\n\n## Findings\n- Two.",
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
        # evidence is union(community_reports_packed, chunk_text_packed)
        # — at least one of the two should be non-empty for this input.
        assert len(result.retrieved_evidence_sentences) >= 1
        # Community-report sections produced by the scripted answerer
        # carry the "## Title" marker; the chunk text carries "Alice".
        joined = "\n".join(result.retrieved_evidence_sentences)
        assert "## Title" in joined or "Alice" in joined
