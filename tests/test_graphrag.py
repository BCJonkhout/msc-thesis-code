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

from pilot.architectures import graphrag as graphrag_mod
from pilot.architectures.graphrag import (
    _ENTITY_EXTRACT_PROMPT,
    _ENTITY_GLEAN_PROMPT,
    _Entity,
    _merge_extraction,
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


# ──────────────────────────────────────────────────────────────────────
# Entity-extraction prompt structure
# ──────────────────────────────────────────────────────────────────────

class TestEntityExtractPromptStructure:
    """The extraction prompt is the load-bearing knob for
    extraction recall on small models (Flash Lite). The audit at
    `thesis-msc/notes/paper_implementation_audit.md` 156-165 flagged
    the prior no-few-shot, strict-JSON prompt as the most likely
    cause of under-reported GraphRAG F1. These assertions pin the
    ported prompt against silent regressions: three few-shot
    examples (Microsoft's count), open entity-type rule preserved,
    relationship-strength field present, JSON output shape kept so
    the downstream parser doesn't need its own port.
    """

    def test_prompt_contains_three_few_shot_examples(self):
        # Microsoft's GRAPH_EXTRACTION_PROMPT pattern uses
        # "Example 1", "Example 2", "Example 3" as section markers.
        for marker in ("Example 1", "Example 2", "Example 3"):
            assert marker in _ENTITY_EXTRACT_PROMPT, (
                f"missing few-shot marker {marker!r}; the audit flags "
                f"three shots as load-bearing for small-model extraction"
            )
        # Make sure we don't silently grow a fourth shot (Microsoft
        # caps at three; more would crowd the chunk out of the
        # context window on smaller models).
        assert "Example 4" not in _ENTITY_EXTRACT_PROMPT

    def test_prompt_preserves_open_entity_type_extraction(self):
        # The audit (lines 178-183) calls out open extraction as
        # deliberate. The prompt must NOT hard-code Microsoft's
        # four-type list and must explicitly tell the model to
        # choose types freely.
        forbidden_fixed_type_list = (
            "organization, person, geo, event",
            "ORGANIZATION,PERSON,GEO,EVENT",
        )
        for phrase in forbidden_fixed_type_list:
            assert phrase not in _ENTITY_EXTRACT_PROMPT
        # And the prompt must explicitly invite open extraction.
        assert "do not constrain" in _ENTITY_EXTRACT_PROMPT.lower() or \
               "do not restrict" in _ENTITY_EXTRACT_PROMPT.lower() or \
               "choose freely" in _ENTITY_EXTRACT_PROMPT.lower()

    def test_prompt_keeps_strict_json_output_shape(self):
        # Downstream `_parse_extract_json` expects a JSON object with
        # `entities` and `relationships` arrays. The schema line in
        # the prompt must match that shape (preserved verbatim so
        # the parser doesn't need a parallel port).
        assert '"entities"' in _ENTITY_EXTRACT_PROMPT
        assert '"relationships"' in _ENTITY_EXTRACT_PROMPT
        assert '"name"' in _ENTITY_EXTRACT_PROMPT
        assert '"source"' in _ENTITY_EXTRACT_PROMPT
        assert '"target"' in _ENTITY_EXTRACT_PROMPT
        # The Microsoft delimited-tuple format must NOT leak through
        # — if it did, parsing would silently fall back to the empty
        # extraction.
        assert "<|>" not in _ENTITY_EXTRACT_PROMPT
        assert "<|COMPLETE|>" not in _ENTITY_EXTRACT_PROMPT

    def test_prompt_includes_relationship_strength(self):
        # Microsoft's `relationship_strength` 1-10 score is preserved
        # as a `weight` field in our JSON shape; the prompt must
        # instruct the model to emit it.
        assert '"weight"' in _ENTITY_EXTRACT_PROMPT
        # The instruction language should reference the 1-10 scale.
        assert "1-10" in _ENTITY_EXTRACT_PROMPT or "1 to 10" in _ENTITY_EXTRACT_PROMPT

    def test_prompt_examples_cover_both_workloads(self):
        # QASPER (research-paper) markers: the audit (line 178-183)
        # singles out methods, datasets, models, metrics as the
        # entities our open extraction must surface.
        lower = _ENTITY_EXTRACT_PROMPT.lower()
        for marker in ("qasper", "f1"):
            assert marker in lower, f"research-paper few-shot missing {marker!r}"
        # NovelQA (narrative fiction) markers: at least one shot
        # should be a narrative passage with characters and a
        # location, not a research-paper paragraph.
        assert "character" in lower
        # The narrative shot in the seed file uses Pemberley as the
        # location name; if that example is replaced, this assertion
        # should be updated alongside it.
        assert "Pemberley" in _ENTITY_EXTRACT_PROMPT

    def test_glean_prompt_aligns_with_extract_prompt(self):
        # The gleaning re-prompt must reuse the same JSON output
        # shape and the same open-type rule, otherwise the model
        # could silently switch formats between passes and the
        # parser would drop the gleaning yield.
        assert "JSON" in _ENTITY_GLEAN_PROMPT or "json" in _ENTITY_GLEAN_PROMPT
        assert '"entities"' in _ENTITY_GLEAN_PROMPT
        assert '"relationships"' in _ENTITY_GLEAN_PROMPT

    def test_few_shot_example_outputs_parse_under_current_parser(self):
        """Each few-shot example's `Output:` block must parse cleanly
        under `_parse_extract_json`. If a future prompt edit breaks
        the example JSON (trailing comma, typo, fenced code) the
        model will be shown an unparseable target and our downstream
        extraction will silently degrade. Regression-test the three
        seed examples against the live parser by scanning the
        rendered prompt for `Output:` blocks and feeding each one to
        the parser the runtime uses."""
        rendered = _ENTITY_EXTRACT_PROMPT.format(chunk="<placeholder>")
        # The three few-shot Output blocks all start with `Output:`
        # followed by a JSON object opened on the next line. The
        # last `Output:` in the rendered prompt is the trailing
        # instruction to the model and has no JSON after it — skip
        # it and assert against the leading three.
        segments = rendered.split("Output:")
        parsed_with_entities = 0
        for segment in segments[1:]:  # skip pre-first-Output prose
            # Look at the first top-level JSON object in this
            # segment. Find the first `{` and walk to its matching
            # `}` with brace counting.
            start = segment.find("{")
            if start == -1:
                continue
            depth = 0
            end = -1
            for i, ch in enumerate(segment[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                continue
            block = segment[start:end]
            try:
                out = _parse_extract_json(block)
            except Exception:
                continue
            if out.get("entities"):
                parsed_with_entities += 1
        assert parsed_with_entities >= 3, (
            f"expected at least 3 few-shot Output blocks to parse as "
            f"populated entity extractions; got {parsed_with_entities}"
        )


# ──────────────────────────────────────────────────────────────────────
# _merge_extraction — relationship weight / strength
# ──────────────────────────────────────────────────────────────────────

class TestMergeExtractionRelationshipStrength:
    """Microsoft graphrag's `relationship_strength` 1-10 score is
    preserved as a `weight` field in our JSON shape; the merge must
    accept multiple aliases (`weight`, `strength`, `relationship_strength`)
    and clamp pathological values.
    """

    def test_weight_field_populates_strength_on_relationship(self):
        ents: dict[str, _Entity] = {}
        rels: list[_Relationship] = []
        parsed = {
            "entities": [{"name": "A", "type": "x", "description": ""}],
            "relationships": [
                {"source": "A", "target": "B", "description": "r", "weight": 9},
            ],
        }
        _merge_extraction(ents, rels, parsed, chunk_idx=0)
        assert len(rels) == 1
        assert rels[0].strength == 9

    def test_microsoft_alias_relationship_strength_is_accepted(self):
        ents: dict[str, _Entity] = {}
        rels: list[_Relationship] = []
        parsed = {
            "entities": [],
            "relationships": [
                {"source": "A", "target": "B", "description": "r",
                 "relationship_strength": 7},
            ],
        }
        _merge_extraction(ents, rels, parsed, chunk_idx=0)
        assert rels[0].strength == 7

    def test_pathological_strength_is_clamped(self):
        ents: dict[str, _Entity] = {}
        rels: list[_Relationship] = []
        parsed = {
            "entities": [],
            "relationships": [
                {"source": "A", "target": "B", "description": "r", "weight": 999},
                {"source": "C", "target": "D", "description": "r", "weight": -5},
            ],
        }
        _merge_extraction(ents, rels, parsed, chunk_idx=0)
        assert rels[0].strength == 10
        assert rels[1].strength == 1

    def test_missing_strength_leaves_field_none(self):
        ents: dict[str, _Entity] = {}
        rels: list[_Relationship] = []
        parsed = {
            "entities": [],
            "relationships": [
                {"source": "A", "target": "B", "description": "r"},
            ],
        }
        _merge_extraction(ents, rels, parsed, chunk_idx=0)
        assert rels[0].strength is None

    def test_strength_appears_as_edge_attribute_after_graph_build(self):
        # Two extractions of the same pair with different strengths
        # should average to the midpoint on the resulting edge.
        ents = [
            _Entity(name="A", type="x", description=""),
            _Entity(name="B", type="x", description=""),
        ]
        rels = [
            _Relationship(source="A", target="B", description="r1",
                          text_unit_id=0, strength=10),
            _Relationship(source="A", target="B", description="r2",
                          text_unit_id=1, strength=2),
        ]
        g, _communities, _edge_tu = _build_graph_and_communities(ents, rels)
        edge = g.edges["A", "B"]
        # weight is still chunk co-occurrence count (preserved
        # behaviour); strength is the mean of per-extraction
        # strengths so downstream code can break ties between
        # equally-weighted edges by tight-vs-loose relationship.
        assert edge["weight"] == 2
        assert edge["strength"] == 6.0
