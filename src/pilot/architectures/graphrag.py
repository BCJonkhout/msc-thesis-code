"""GraphRAG runner — faithful from-scratch local-search implementation.

Implements Edge et al. 2024 §3.1 + Microsoft graphrag's local_search
(`packages/graphrag/graphrag/query/structured_search/local_search/`)
against the pilot's primitives — `OllamaEmbedder`, `AnswererProvider`,
`CostLedger`, `SentenceBoundaryChunker` — instead of pulling in
Microsoft's `graphrag` package and its multi-class `LLMCompletion` /
`LLMEmbedding` shim layer.

Pipeline (per Edge et al. §3.1 + Microsoft `LocalSearchMixedContext`):

  1. Chunk source documents at 600 / 100 (paper §4.1.1).
  2. Per-chunk LLM call extracting entities + relationships. Entities
     keep a list of ``text_unit_ids`` tracking which chunks they
     appear in; relationships gain a ``weight`` = co-occurring-chunk
     count.
  3. One gleaning pass per chunk (Microsoft default `max_gleanings=1`):
     re-run extraction asking the LLM "did you miss anything?". This
     materially improves recall at the cost of one extra LLM call
     per chunk.
  4. Build NetworkX graph: nodes = entities, edges = relationships.
  5. Louvain community detection (NetworkX 3.x). Deliberate
     simplification vs hierarchical Leiden via graspologic — both
     are modularity-optimising; results differ in cross-run
     stability rather than in static-document quality.
  6. Per-community LLM call producing a 500–700-token structured
     report (title / summary / findings) — replacing the prior
     2–4-sentence summaries which discarded too much information for
     entity-specific QA.
  7. Local search at query time, per
     `LocalSearchMixedContext.build_context`:

        a. Embed query + all entity descriptions (cached for the
           document).
        b. Top-`top_k_entities` entities by query-entity-description
           cosine (default 10, paper-aligned).
        c. For each selected entity, transitively gather:
             - Communities the entity belongs to (sorted by
               match-count desc, then community size desc).
             - Top-`top_k_relationships` relationships by weight.
             - Text units (chunks) the entity appears in, ranked by
               co-occurring-relationship count.
        d. Pack context within `max_context_tokens` (default 8000)
           split:
             - community 0.15
             - text-units 0.50
             - entities + relationships 0.35

  8. Single answerer call against the packed local-search context.

Cost-accounting: every LLM and embedding call writes a CostLedger row.
The pilot's deployment-cost rule (Option A, run_index=0) applies
uniformly.

Deliberate simplifications vs Microsoft `graphrag` package
----------------------------------------------------------

  - **Louvain instead of hierarchical Leiden.** NetworkX 3.x ships
    Louvain natively; graspologic Leiden adds a heavy native
    dependency. Modularity optimisation is the load-bearing
    behaviour; both algorithms produce communities that work for
    local search.
  - **No covariates / claims extraction.** Microsoft's package
    skips this by default too.
  - **No description-summary LLM call per heavily-mentioned
    entity.** Microsoft's package summarises entity descriptions
    when concatenated length exceeds a threshold; we keep raw
    concatenated descriptions, which preserves verbatim factual
    strings (helpful for QASPER's specific-factual queries).
  - **Single-prompt structured community report (markdown),** not
    JSON-schema. Microsoft emits a JSON object with fields
    `title / summary / rating / rating_explanation / findings[]`;
    we emit the same content as markdown headers because robust
    JSON-schema parsing across heterogeneous answerers (Gemini,
    Grok, DeepSeek) is its own multi-day reliability project.

These deviations are documented in the module-level audit row in
`thesis-msc/notes/pilot_findings.md` so reviewers can see the delta
between this pilot's GraphRAG and the published method.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Pin numba/OpenMP threads before networkx and friends import them.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from pilot.architectures.base import ArchitectureResult, _render_prompt
from pilot.encoders import OllamaEmbedder, SentenceBoundaryChunker
from pilot.ledger import CostLedger, Stage, sha256_hex
from pilot.providers.base import AnswererProvider, CacheControl


# ──────────────────────────────────────────────────────────────────────
# Paper-default parameters (Edge et al. 2024 + Microsoft graphrag
# `LocalSearchDefaults`)
# ──────────────────────────────────────────────────────────────────────

_CHUNK_TOKENS = 600   # Edge et al. §4.1.1
_CHUNK_OVERLAP_TOKENS = 100
_TOP_K_ENTITIES = 10  # Microsoft `LocalSearchDefaults.top_k_mapped_entities`
_TOP_K_RELATIONSHIPS = 10  # `LocalSearchDefaults.top_k_relationships`
_MAX_GLEANINGS = 1     # `ExtractGraphDefaults.max_gleanings`
_COMMUNITY_SEED = 0xDEADBEEF

# Local-search context budget (Microsoft default = 12_000 with a
# `max_context_tokens` budget; we use 8_000 to keep prompt costs
# tractable on the QASPER 20Q calibration sweep — the documents
# themselves are 5-15k tokens after chunking, so the budget is
# rarely the binding constraint at this scale).
_MAX_CONTEXT_TOKENS = 8_000
_COMMUNITY_PROP = 0.15  # `LocalSearchDefaults.community_prop`
_TEXT_UNIT_PROP = 0.50  # `LocalSearchDefaults.text_unit_prop`
# remaining 0.35 for entities + relationships


# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────

# Entity-extraction prompt — adapted from Microsoft graphrag
# `prompts/index/extract_graph.py::GRAPH_EXTRACTION_PROMPT`. Two structural
# changes vs the upstream prompt:
#
#   1. **Open entity-type extraction.** Microsoft hard-codes a fixed
#      `entity_types` list (default `[organization, person, geo, event]`)
#      appropriate for the Podcast / News workloads the package was tuned
#      on. Those four types miss the load-bearing entities for QASPER
#      research papers (concepts, methods, datasets, models, metrics) and
#      for NovelQA narratives (characters, locations, plot events,
#      thematic objects). The workload-specific deviation is documented
#      in `thesis-msc/notes/paper_implementation_audit.md` (audit row
#      178-183) as deliberate.
#   2. **Strict-JSON output instead of delimited tuples.** Microsoft
#      emits `("entity"<|>NAME<|>TYPE<|>DESC)` records separated by
#      `##`, terminated by `<|COMPLETE|>`. We keep the JSON output shape
#      our parser (`_parse_extract_json`) and downstream merge already
#      consume — porting the tuple format would force a parallel parser
#      rewrite for no extraction-quality gain. Microsoft's
#      `relationship_strength` 1-10 score is preserved as an optional
#      `weight` field in the JSON; `_merge_extraction` reads it when
#      present and otherwise falls back to the chunk-co-occurrence
#      weight assigned at graph-build time.
#
# What IS ported verbatim from Microsoft: the Goal / Steps / Examples
# scaffolding, the three-shot example structure (which the audit flags
# as load-bearing for small-model extraction quality), and the
# step-by-step instruction to (a) identify entities, (b) identify
# pairs that are clearly related, (c) emit a relationship-strength
# score. The three shots are adapted to the QASPER + NovelQA workload
# mix: one research-paper methods chunk (QASPER-like), one narrative
# fiction passage (NovelQA-like), one technical results paragraph
# combining numerical results and citations (QASPER-like).
_ENTITY_EXTRACT_PROMPT = """-Goal-
Given a document chunk, identify all entities of any type that are relevant to the chunk's content, then identify all pairs of those entities that are clearly related. Do not restrict yourself to a fixed list of types — the workload includes research papers (where concepts, methods, datasets, models, and metrics are load-bearing) and novel-length fiction (where characters, locations, plot events, and thematic objects are load-bearing). Choose whichever entity types best describe what the chunk is actually about.

-Steps-
1. Identify all entities. For each identified entity, extract:
- name: Name of the entity, exactly as it appears in the text (preserve original casing for proper nouns; lowercase for common-noun concepts).
- type: A concise lowercase noun describing the entity category (examples: person, character, location, organisation, dataset, model, method, metric, concept, event, work, object). Choose freely; do not constrain yourself to a closed list.
- description: One sentence describing the entity's attributes and role in the chunk. Preserve verbatim numerical results, model names, and other specific terms.

2. From the entities identified in step 1, identify all pairs of (source, target) that are *clearly related* to each other in the chunk. For each related pair, extract:
- source: name of the source entity, exactly as in step 1.
- target: name of the target entity, exactly as in step 1.
- description: one-sentence explanation of why the two entities are related, grounded in the chunk.
- weight: integer 1-10 indicating relationship strength. Use 8-10 for tight, explicit relationships (X is Y, X causes Y, X reports the result on Y); 4-7 for clear contextual relationships (X is mentioned alongside Y, X compares against Y); 1-3 for incidental co-occurrence.

3. Return output as STRICT JSON with two top-level keys, `entities` and `relationships`, matching this schema:

{{"entities": [{{"name": "...", "type": "...", "description": "..."}}, ...],
 "relationships": [{{"source": "...", "target": "...", "description": "...", "weight": 5}}, ...]}}

If the chunk contains no extractable entities, return {{"entities": [], "relationships": []}}.

######################
-Examples-
######################

Example 1 (research-paper methods chunk, QASPER-style).
Chunk: "We evaluate on the QASPER dataset using GPT-4 with retrieval-augmented generation. The Lewis et al. 2020 RAG model serves as our baseline; we measure token-level F1 across 1,000 held-out questions."
Output:
{{"entities": [
    {{"name": "QASPER", "type": "dataset", "description": "Question-answering dataset over scientific papers used as the evaluation benchmark in this work."}},
    {{"name": "GPT-4", "type": "model", "description": "Large language model used for answer generation, paired with retrieval-augmented generation."}},
    {{"name": "retrieval-augmented generation", "type": "method", "description": "Retrieval-paired generation technique applied to GPT-4 in this evaluation."}},
    {{"name": "Lewis et al. 2020 RAG", "type": "method", "description": "Baseline RAG model from Lewis et al. 2020, used as the comparison point for GPT-4."}},
    {{"name": "token-level F1", "type": "metric", "description": "Evaluation metric measured across 1,000 held-out QASPER questions."}}
  ],
 "relationships": [
    {{"source": "GPT-4", "target": "retrieval-augmented generation", "description": "GPT-4 is paired with retrieval-augmented generation in this evaluation.", "weight": 9}},
    {{"source": "GPT-4", "target": "QASPER", "description": "GPT-4 is evaluated on the QASPER dataset.", "weight": 8}},
    {{"source": "Lewis et al. 2020 RAG", "target": "GPT-4", "description": "Lewis et al. 2020 RAG is the baseline against which GPT-4 is compared.", "weight": 8}},
    {{"source": "token-level F1", "target": "QASPER", "description": "Token-level F1 is the metric reported on QASPER's held-out questions.", "weight": 7}}
  ]}}

Example 2 (narrative fiction passage, NovelQA-style).
Chunk: "Elizabeth glanced once more at her sister, then turned away from the window. Mr Darcy stood at the far end of the drawing-room at Pemberley, conversing with Colonel Fitzwilliam in a low voice. She felt, for the first time that morning, that she might never understand what passed between them."
Output:
{{"entities": [
    {{"name": "Elizabeth", "type": "character", "description": "Female protagonist who observes Mr Darcy and Colonel Fitzwilliam at Pemberley and reflects on her own incomprehension."}},
    {{"name": "Mr Darcy", "type": "character", "description": "Stands at the far end of the drawing-room at Pemberley, conversing in a low voice with Colonel Fitzwilliam."}},
    {{"name": "Colonel Fitzwilliam", "type": "character", "description": "Converses privately with Mr Darcy in the drawing-room at Pemberley."}},
    {{"name": "Pemberley", "type": "location", "description": "Estate whose drawing-room is the setting of this scene."}},
    {{"name": "the drawing-room", "type": "location", "description": "Room at Pemberley where Mr Darcy and Colonel Fitzwilliam are conversing in a low voice."}},
    {{"name": "Elizabeth's sister", "type": "character", "description": "Present in the scene; Elizabeth glances at her before turning away from the window."}}
  ],
 "relationships": [
    {{"source": "Mr Darcy", "target": "Colonel Fitzwilliam", "description": "Mr Darcy and Colonel Fitzwilliam are conversing in a low voice in the drawing-room.", "weight": 9}},
    {{"source": "Mr Darcy", "target": "Pemberley", "description": "Mr Darcy is at Pemberley in this scene.", "weight": 8}},
    {{"source": "Colonel Fitzwilliam", "target": "Pemberley", "description": "Colonel Fitzwilliam is at Pemberley in this scene.", "weight": 8}},
    {{"source": "Elizabeth", "target": "Pemberley", "description": "Elizabeth is present at Pemberley, observing Mr Darcy and Colonel Fitzwilliam.", "weight": 7}},
    {{"source": "Elizabeth", "target": "Mr Darcy", "description": "Elizabeth watches Mr Darcy and reflects that she may never understand what passes between him and Colonel Fitzwilliam.", "weight": 7}},
    {{"source": "Elizabeth", "target": "Elizabeth's sister", "description": "Elizabeth glances at her sister before turning away from the window.", "weight": 5}}
  ]}}

Example 3 (technical results paragraph, QASPER-style).
Chunk: "Naive RAG performs the worst, achieving 0.137 F1, while RAPTOR scores 0.267 on the QASPER calibration pool. GraphRAG (Edge et al. 2024) lands at 0.220 F1, between the two. All three architectures use BGE-M3 for retrieval embeddings and Gemini Flash Lite as the answerer."
Output:
{{"entities": [
    {{"name": "Naive RAG", "type": "method", "description": "Baseline RAG architecture achieving 0.137 F1 on the QASPER calibration pool, the lowest of the three compared."}},
    {{"name": "RAPTOR", "type": "method", "description": "Tree-based retrieval method achieving 0.267 F1 on the QASPER calibration pool, the highest of the three compared."}},
    {{"name": "GraphRAG", "type": "method", "description": "Graph-based retrieval-augmented generation method from Edge et al. 2024, achieving 0.220 F1 on the QASPER calibration pool."}},
    {{"name": "Edge et al. 2024", "type": "citation", "description": "Reference for the GraphRAG method."}},
    {{"name": "QASPER calibration pool", "type": "dataset", "description": "Subset of QASPER used to calibrate the three architectures."}},
    {{"name": "F1", "type": "metric", "description": "Token-level F1 used to score architecture outputs on the QASPER calibration pool."}},
    {{"name": "BGE-M3", "type": "model", "description": "Retrieval embedding model shared by all three architectures."}},
    {{"name": "Gemini Flash Lite", "type": "model", "description": "Answerer model shared by all three architectures."}}
  ],
 "relationships": [
    {{"source": "Naive RAG", "target": "QASPER calibration pool", "description": "Naive RAG is evaluated on the QASPER calibration pool and achieves 0.137 F1.", "weight": 9}},
    {{"source": "RAPTOR", "target": "QASPER calibration pool", "description": "RAPTOR is evaluated on the QASPER calibration pool and achieves 0.267 F1.", "weight": 9}},
    {{"source": "GraphRAG", "target": "QASPER calibration pool", "description": "GraphRAG is evaluated on the QASPER calibration pool and achieves 0.220 F1.", "weight": 9}},
    {{"source": "GraphRAG", "target": "Edge et al. 2024", "description": "GraphRAG is the method introduced in Edge et al. 2024.", "weight": 10}},
    {{"source": "Naive RAG", "target": "RAPTOR", "description": "Both methods are compared on the same QASPER pool; RAPTOR scores higher.", "weight": 6}},
    {{"source": "Naive RAG", "target": "GraphRAG", "description": "Both methods are compared on the same QASPER pool; GraphRAG scores higher.", "weight": 6}},
    {{"source": "RAPTOR", "target": "GraphRAG", "description": "Both methods are compared on the same QASPER pool; RAPTOR scores higher.", "weight": 6}},
    {{"source": "BGE-M3", "target": "Naive RAG", "description": "BGE-M3 supplies retrieval embeddings for Naive RAG.", "weight": 7}},
    {{"source": "BGE-M3", "target": "RAPTOR", "description": "BGE-M3 supplies retrieval embeddings for RAPTOR.", "weight": 7}},
    {{"source": "BGE-M3", "target": "GraphRAG", "description": "BGE-M3 supplies retrieval embeddings for GraphRAG.", "weight": 7}},
    {{"source": "Gemini Flash Lite", "target": "Naive RAG", "description": "Gemini Flash Lite is the answerer for Naive RAG.", "weight": 7}},
    {{"source": "Gemini Flash Lite", "target": "RAPTOR", "description": "Gemini Flash Lite is the answerer for RAPTOR.", "weight": 7}},
    {{"source": "Gemini Flash Lite", "target": "GraphRAG", "description": "Gemini Flash Lite is the answerer for GraphRAG.", "weight": 7}},
    {{"source": "F1", "target": "QASPER calibration pool", "description": "F1 is the metric reported on the QASPER calibration pool.", "weight": 8}}
  ]}}

######################
-Real Data-
######################

Now extract entities and relationships from the following chunk. Return STRICT JSON only — no preamble, no trailing commentary, no markdown code fences.

Document chunk:
{chunk}

Output:
"""

# Microsoft graphrag's gleaning loop (CONTINUE_PROMPT in
# `prompts/index/extract_graph.py`) re-prompts the model to emit any
# entities or relationships missed in the first pass. We follow the
# same single-call pattern, in the same JSON shape as the initial
# extraction, so `_parse_extract_json` + `_merge_extraction` need no
# additional branching.
_ENTITY_GLEAN_PROMPT = """MANY entities and relationships were missed in the last extraction from this chunk. Re-examine the chunk and emit any ADDITIONAL entities and relationships that were not in your prior response. Use the same strict-JSON format, the same open-entity-type rule (no fixed type list), and the same 1-10 relationship weight scale.

If you found nothing new, return {{"entities": [], "relationships": []}}.

Document chunk:
{chunk}

Output:
"""

# Lengthened, structured community-report prompt (markdown sections
# substituting for Microsoft's JSON schema). Target ~500-700 tokens
# per report — closer to the Microsoft package's 2000-token cap than
# our previous 2-4-sentence summaries which discarded the verbatim
# detail QASPER's questions need.
_COMMUNITY_REPORT_PROMPT = """You are writing a community report on a thematic group of entities extracted from a document. The report will be used by a question-answering system that retrieves it when its query touches one of these entities.

The community contains the following entities and pairwise relationships.

# Entities
{entities}

# Relationships
{relationships}

Write a structured markdown report with the following sections, preserving verbatim factual details from the entity descriptions wherever possible:

## Title
A 4-12 word title naming the community by its dominant theme or central entities.

## Summary
A 3-5 sentence summary covering: who/what is in this community, what role they play in the document, and the most important relationships among them. Preserve specific names, numbers, and technical terms verbatim.

## Findings
3-7 bullet points stating concrete facts about the entities and relationships. Each finding should be a complete factual sentence drawn from the entity descriptions, not a meta-summary. Numbers, model names, dataset names, methods, and other specific terms must be preserved verbatim.

Respond with the markdown report only. No preamble or trailing commentary.
"""


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _Entity:
    name: str
    type: str
    description: str
    text_unit_ids: list[int] = field(default_factory=list)


@dataclass
class _Relationship:
    source: str
    target: str
    description: str
    text_unit_id: int = -1
    # Optional per-extraction relationship-strength score (Microsoft
    # graphrag's `relationship_strength`, 1-10). Parsed lazily from the
    # LLM's JSON output when present; left as `None` when the model
    # omits it. The chunk-co-occurrence count assigned at graph-build
    # time remains the primary edge-weight signal — `strength` is
    # treated as an additive bonus so missing-strength extractions
    # still produce a valid graph.
    strength: int | None = None


@dataclass
class _CommunityReport:
    community_id: int
    member_names: list[str]
    text: str          # rendered markdown report
    rank: int          # community size (proxy for Microsoft's rank attribute)


@dataclass
class _GraphRAGState:
    """Per-(run, paper) preprocessing artefact for GraphRAG.

    Carries the full output of steps 1–4 (chunking, entity extraction
    with gleaning, graph build + Louvain communities, per-community
    structured reports) PLUS the entity-description embeddings (step
    4.5), so subsequent questions on the same paper only pay the
    query-embed + local-search-pack + answerer cost. Without this
    cache the ledger over-attributes ``C_off^struct`` by a factor of N
    (questions per document), which would silently break the
    repeated-context amortisation that motivates the whole study.

    Entity-description embeddings are stored here (rather than
    recomputed inside ``_local_search_build_context`` on every
    question) because they are conceptually part of the index
    build — Microsoft graphrag persists them to a vector store at
    indexing time — and re-embedding them per question would both
    waste wall-clock (hundreds of redundant BGE-M3 calls per paper)
    and mis-attribute index-build cost to per-query retrieval.

    Pickle contract
    ---------------
    Every field here is pure data: lists of dataclass records, a
    NetworkX ``Graph`` (whose own state is a node-attr / edge-attr
    dict), nested str/int/float. No httpx clients, no provider
    handles. Pickle round-trips trivially, but ``__getstate__`` /
    ``__setstate__`` / ``rehydrate`` are defined for symmetry with
    ``pilot.architectures.raptor._RaptorState`` — any future addition
    of a live adapter field would need to mirror the strip-on-save
    pattern, and a uniform ``state.rehydrate(...)`` call on the
    cache-loading side keeps the consumer code arch-agnostic.

    Byte-equivalence guarantee: two candidates that load the same
    pickle produce the same generate-stage ``prompt_hash`` for the
    same query because all retrieval inputs (entity vectors,
    community reports, packed-context budget split) come from the
    cached state, and the only per-query embed call (the query
    itself) is deterministic across candidates that share an
    encoder.
    """
    chunks: list[str]
    entities: list["_Entity"]
    relationships: list["_Relationship"]
    g: object  # NetworkX Graph
    communities: list[set[str]]
    reports: list["_CommunityReport"]
    entity_vecs: list[list[float]]
    embed_dim: int | None

    def __getstate__(self) -> dict:
        # No live-adapter fields to strip today, but the explicit
        # __getstate__ pins the on-disk schema so a future drift into
        # holding an embedder / provider handle here is caught at
        # review rather than silently re-introducing the pickle bug
        # ``_RaptorState`` already had.
        return {
            "chunks": self.chunks,
            "entities": self.entities,
            "relationships": self.relationships,
            "g": self.g,
            "communities": self.communities,
            "reports": self.reports,
            "entity_vecs": self.entity_vecs,
            "embed_dim": self.embed_dim,
        }

    def __setstate__(self, state: dict) -> None:
        self.chunks = state["chunks"]
        self.entities = state["entities"]
        self.relationships = state["relationships"]
        self.g = state["g"]
        self.communities = state["communities"]
        self.reports = state["reports"]
        self.entity_vecs = state["entity_vecs"]
        self.embed_dim = state.get("embed_dim")

    def rehydrate(
        self,
        *,
        embedder: object = None,  # noqa: ARG002 — symmetry stub
        ledger: object = None,    # noqa: ARG002
        answerer: object = None,  # noqa: ARG002
        answerer_model: str | None = None,  # noqa: ARG002
        summary_answerer: object = None,    # noqa: ARG002
        summary_model: str | None = None,   # noqa: ARG002
        run_index: int = 0,                  # noqa: ARG002
        max_tokens: int = 256,               # noqa: ARG002
    ) -> "_GraphRAGState":
        """No-op rehydrate, present for symmetry with ``_RaptorState``.

        GraphRAG's local search is constructed per-question from
        free-standing helpers in this module that take the live
        embedder / ledger / answerer as positional arguments; nothing
        on the state object itself needs re-wiring. The signature
        mirrors ``_RaptorState.rehydrate`` so callers can rehydrate
        either arch's state with the same kwargs.
        """
        return self


# ──────────────────────────────────────────────────────────────────────
# JSON parsing helpers
# ──────────────────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def _parse_extract_json(raw: str) -> dict:
    """Lenient JSON extractor for entity-extraction LLM output."""
    if not raw or not raw.strip():
        return {"entities": [], "relationships": []}
    try:
        return json.loads(raw)
    except Exception:
        pass
    candidates = _JSON_BLOCK_RE.findall(raw)
    for c in sorted(candidates, key=len, reverse=True):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and ("entities" in obj or "relationships" in obj):
                return obj
        except Exception:
            continue
    return {"entities": [], "relationships": []}


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def _approx_token_count(text: str) -> int:
    """Cheap char/4 token approximation. The pilot uses this everywhere
    a tokenizer would be overkill (cost ledger fallback, retrieval
    budget packing). Within ~10% of OpenAI tokenisers for English text."""
    return max(1, len(text) // 4)


# ──────────────────────────────────────────────────────────────────────
# Entity extraction (with gleaning)
# ──────────────────────────────────────────────────────────────────────

def _merge_extraction(
    entities: dict[str, _Entity],
    relationships: list[_Relationship],
    parsed: dict,
    *,
    chunk_idx: int,
) -> None:
    """In-place merge of one parsed extraction into the cumulative
    entity dict + relationship list. Tracks which chunks each entity
    appeared in via text_unit_ids."""
    for raw_e in parsed.get("entities", []) or []:
        if not isinstance(raw_e, dict):
            continue
        name = (raw_e.get("name") or "").strip()
        if not name:
            continue
        etype = (raw_e.get("type") or "").strip() or "unknown"
        desc = (raw_e.get("description") or "").strip()
        existing = entities.get(name)
        if existing is None:
            entities[name] = _Entity(
                name=name, type=etype, description=desc,
                text_unit_ids=[chunk_idx],
            )
        else:
            if desc and desc not in existing.description:
                existing.description = (
                    f"{existing.description}\n{desc}"
                    if existing.description else desc
                )
            if chunk_idx not in existing.text_unit_ids:
                existing.text_unit_ids.append(chunk_idx)

    for raw_r in parsed.get("relationships", []) or []:
        if not isinstance(raw_r, dict):
            continue
        src = (raw_r.get("source") or "").strip()
        tgt = (raw_r.get("target") or "").strip()
        if not src or not tgt:
            continue
        desc = (raw_r.get("description") or "").strip()
        # Microsoft graphrag emits `relationship_strength` as an integer
        # 1-10. We accept either `weight` (our JSON shape) or
        # `strength` / `relationship_strength` (Microsoft-style) for
        # forward-compatibility if a future prompt switch reuses the
        # upstream key names. Lenient coercion: any non-integer falls
        # back to `None` so the chunk-co-occurrence weight remains the
        # primary signal.
        raw_strength = (
            raw_r.get("weight")
            if raw_r.get("weight") is not None
            else raw_r.get("strength")
            if raw_r.get("strength") is not None
            else raw_r.get("relationship_strength")
        )
        strength: int | None = None
        if raw_strength is not None:
            try:
                strength = int(round(float(raw_strength)))
                # Clamp to Microsoft's documented 1-10 range so a
                # pathological LLM output (e.g. 99) can't dominate
                # every other edge in the graph.
                strength = max(1, min(10, strength))
            except (TypeError, ValueError):
                strength = None
        relationships.append(_Relationship(
            source=src, target=tgt, description=desc,
            text_unit_id=chunk_idx, strength=strength,
        ))


def _extract_entities_per_chunk(
    chunks: list[str],
    *,
    answerer: AnswererProvider,
    answerer_model: str,
    ledger: CostLedger,
    run_index: int,
    max_tokens: int = 1024,
    max_gleanings: int = _MAX_GLEANINGS,
) -> tuple[list[_Entity], list[_Relationship]]:
    """One LLM call per chunk plus ``max_gleanings`` re-extraction passes.

    Microsoft graphrag's `extract_graph` does the same: initial
    extraction, then up to `max_gleanings` follow-up calls per chunk
    asking the model to emit anything it missed. The default is 1.
    """
    entities: dict[str, _Entity] = {}
    relationships: list[_Relationship] = []

    # Each (chunk, pass) extraction call depends only on its chunk text (the
    # gleaning prompt re-reads the chunk; it does not chain on the prior
    # pass), so the calls are independent and can run concurrently. The cost
    # bottleneck of a GraphRAG build is exactly these per-chunk provider calls
    # (one initial + max_gleanings per chunk), so overlapping them is the main
    # build speed-up on long documents.
    #
    # Determinism is preserved by separating the work into (1) a concurrent
    # FETCH of every (chunk, pass) extraction and (2) a SEQUENTIAL merge in
    # fixed (chunk_idx, pass_idx) order — identical to the original loop's
    # merge order regardless of which calls finish first. Concurrency is
    # bounded by PILOT_BUILD_CONCURRENCY (default 1 = sequential).
    tasks = [
        (chunk_idx, pass_idx)
        for chunk_idx in range(len(chunks))
        for pass_idx in range(max_gleanings + 1)
    ]

    def _fetch(chunk_idx: int, pass_idx: int) -> dict:
        chunk = chunks[chunk_idx]
        prompt = (
            _ENTITY_EXTRACT_PROMPT.format(chunk=chunk)
            if pass_idx == 0
            else _ENTITY_GLEAN_PROMPT.format(chunk=chunk)
        )
        with ledger.log_call(
            architecture="graphrag",
            stage=Stage.PREPROCESS,
            model=answerer_model,
            prompt=prompt,
            run_index=run_index,
            temperature=0.0,
            max_tokens=max_tokens,
        ) as rec:
            result = answerer.call(
                prompt,
                model=answerer_model,
                max_tokens=max_tokens,
                temperature=0.0,
                cache_control=CacheControl.EPHEMERAL_5MIN,
            )
            rec.uncached_input_tokens = result.uncached_input_tokens
            rec.cached_input_tokens = result.cached_input_tokens
            rec.output_tokens = result.output_tokens
            rec.provider_request_id = result.provider_request_id
            rec.response_hash = sha256_hex(result.text or "")
        return _parse_extract_json(result.text or "")

    build_concurrency = max(1, int(os.environ.get("PILOT_BUILD_CONCURRENCY", "1")))
    parsed_by_task: dict[tuple[int, int], dict] = {}
    if build_concurrency > 1:
        with ThreadPoolExecutor(max_workers=build_concurrency) as executor:
            futures = {
                executor.submit(_fetch, ci, pi): (ci, pi) for ci, pi in tasks
            }
            for fut in futures:
                ci, pi = futures[fut]
                parsed_by_task[(ci, pi)] = fut.result()
    else:
        for ci, pi in tasks:
            parsed_by_task[(ci, pi)] = _fetch(ci, pi)

    # Deterministic merge: fixed (chunk_idx, pass_idx) order, identical to the
    # reference sequential extraction. The early-stop mirrors the original —
    # an empty gleaning pass is not merged (and, for max_gleanings>1, ends
    # that chunk's passes).
    for chunk_idx in range(len(chunks)):
        for pass_idx in range(max_gleanings + 1):
            parsed = parsed_by_task[(chunk_idx, pass_idx)]
            if pass_idx > 0 and not parsed.get("entities") and not parsed.get("relationships"):
                break
            _merge_extraction(entities, relationships, parsed, chunk_idx=chunk_idx)

    return list(entities.values()), relationships


# ──────────────────────────────────────────────────────────────────────
# Graph + community detection
# ──────────────────────────────────────────────────────────────────────

def _build_graph_and_communities(
    entities: list[_Entity],
    relationships: list[_Relationship],
    *,
    seed: int = _COMMUNITY_SEED,
) -> tuple[object, list[set[str]], dict[tuple[str, str], list[int]]]:
    """Construct a NetworkX graph and run Louvain community detection.

    Returns (graph, communities, edge_text_unit_map). The
    edge_text_unit_map records, for each undirected edge key, the list
    of chunk indices the relationship was extracted from (used by local
    search to rank text units).
    """
    import networkx as nx

    g = nx.Graph()
    for e in entities:
        g.add_node(
            e.name,
            type=e.type,
            description=e.description,
            text_unit_ids=tuple(e.text_unit_ids),
        )
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    edge_descriptions: dict[tuple[str, str], list[str]] = defaultdict(list)
    edge_text_units: dict[tuple[str, str], list[int]] = defaultdict(list)
    # Microsoft's `relationship_strength` accumulates per edge so that
    # a tight relationship surfaced once (strength 9) outranks a weak
    # relationship surfaced once (strength 2). When a strength is
    # missing we treat it as the neutral midpoint 5 so legacy un-scored
    # extractions still contribute to the edge weight.
    edge_strength_sum: dict[tuple[str, str], int] = defaultdict(int)
    for r in relationships:
        key = tuple(sorted((r.source, r.target)))
        edge_counts[key] += 1
        edge_strength_sum[key] += r.strength if r.strength is not None else 5
        if r.description:
            edge_descriptions[key].append(r.description)
        edge_text_units[key].append(r.text_unit_id)
        for n in key:
            if n not in g:
                g.add_node(n, type="unknown", description="", text_unit_ids=())
    for key, count in edge_counts.items():
        # Edge weight combines chunk co-occurrence count (primary signal,
        # preserved verbatim) with the mean per-extraction
        # `relationship_strength` scaled to [0, 1]. Doing this
        # multiplicatively keeps the existing local-search ranking
        # behaviour (sort relationships by weight descending) and means
        # an LLM that ignores the strength field (returning every edge
        # at the neutral midpoint 5) reproduces the prior weight=count
        # behaviour exactly. The 0.1 floor prevents a hypothetical
        # all-1 strength from zeroing out an edge that nevertheless
        # appears across many chunks.
        mean_strength = edge_strength_sum[key] / count
        strength_scale = max(0.1, mean_strength / 10.0)
        g.add_edge(
            key[0], key[1],
            weight=count,
            strength=round(mean_strength, 2),
            score=round(count * strength_scale, 3),
            description="\n".join(edge_descriptions[key]),
        )

    if g.number_of_nodes() == 0:
        return g, [], {}

    rng = random.Random(seed)
    communities = nx.community.louvain_communities(g, seed=rng.randint(0, 2**31 - 1))
    return g, communities, dict(edge_text_units)


# ──────────────────────────────────────────────────────────────────────
# Community summarisation
# ──────────────────────────────────────────────────────────────────────

def _summarise_communities(
    g,
    communities: list[set[str]],
    *,
    answerer: AnswererProvider,
    answerer_model: str,
    ledger: CostLedger,
    run_index: int,
    max_tokens: int = 800,
) -> list[_CommunityReport]:
    """Per-community LLM call producing a structured ~500-700 token report.

    Each community's report is an independent, deterministic (T=0) LLM call,
    so the calls run concurrently (bounded by PILOT_BUILD_CONCURRENCY, default
    1 = sequential) while the reports are assembled in fixed community order,
    keeping the build byte-identical to the sequential reference.
    """
    # Build the per-community prompts in deterministic order first.
    jobs: list[tuple[int, list[str], str]] = []
    for community_idx, members in enumerate(communities):
        if not members:
            continue
        member_list = sorted(members)
        ent_lines = []
        for name in member_list:
            data = g.nodes[name]
            ent_lines.append(
                f"- **{name}** (type: {data.get('type', 'unknown')}): "
                f"{data.get('description', '')}"
            )
        rel_lines = []
        for u, v, data in g.edges(member_list, data=True):
            if u in members and v in members:
                rel_lines.append(
                    f"- **{u} ↔ {v}** (weight {data.get('weight', 1)}): "
                    f"{data.get('description', '')}"
                )
        prompt = _COMMUNITY_REPORT_PROMPT.format(
            entities="\n".join(ent_lines),
            relationships="\n".join(rel_lines) or "_(no within-community relationships)_",
        )
        jobs.append((community_idx, member_list, prompt))

    def _report_text(prompt: str) -> str:
        with ledger.log_call(
            architecture="graphrag",
            stage=Stage.PREPROCESS,
            model=answerer_model,
            prompt=prompt,
            run_index=run_index,
            temperature=0.0,
            max_tokens=max_tokens,
        ) as rec:
            result = answerer.call(
                prompt,
                model=answerer_model,
                max_tokens=max_tokens,
                temperature=0.0,
                cache_control=CacheControl.EPHEMERAL_5MIN,
            )
            rec.uncached_input_tokens = result.uncached_input_tokens
            rec.cached_input_tokens = result.cached_input_tokens
            rec.output_tokens = result.output_tokens
            rec.provider_request_id = result.provider_request_id
            rec.response_hash = sha256_hex(result.text or "")
        return result.text or ""

    build_concurrency = max(1, int(os.environ.get("PILOT_BUILD_CONCURRENCY", "1")))
    if build_concurrency > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=build_concurrency) as executor:
            texts = list(executor.map(lambda j: _report_text(j[2]), jobs))
    else:
        texts = [_report_text(prompt) for _, _, prompt in jobs]

    reports: list[_CommunityReport] = []
    for (community_idx, member_list, _), text in zip(jobs, texts):
        reports.append(_CommunityReport(
            community_id=community_idx,
            member_names=member_list,
            text=text,
            rank=len(member_list),
        ))
    return reports


# ──────────────────────────────────────────────────────────────────────
# Local search (Microsoft `LocalSearchMixedContext.build_context`)
# ──────────────────────────────────────────────────────────────────────

def _entity_to_community(
    communities: list[set[str]],
) -> dict[str, list[int]]:
    """Reverse map: entity name → list of community indices it belongs to."""
    out: dict[str, list[int]] = defaultdict(list)
    for c_idx, members in enumerate(communities):
        for name in members:
            out[name].append(c_idx)
    return dict(out)


def _pack_within_budget(
    items: list[tuple[str, int]],
    budget_tokens: int,
) -> list[str]:
    """Greedy token-budget packer. Items are (text, approx_token_count).
    Returns the prefix that fits."""
    used = 0
    out: list[str] = []
    for text, tok in items:
        if used + tok > budget_tokens:
            break
        out.append(text)
        used += tok
    return out


# Heavily-mentioned entities accumulate concatenated descriptions
# across every chunk they appear in; for a long document with many
# chunks an entity description can grow to several thousand
# characters. Truncate each entity-text to a generous prefix — the
# embedding signal is preserved without sending the entire
# multi-thousand-character concatenation through Ollama.
_ENT_DESC_CHAR_CAP = 1000


def _safe_embed_many(
    texts: list[str],
    *,
    embedder: OllamaEmbedder,
    ledger: CostLedger,
    run_index: int,
    stage: Stage,
) -> tuple[list[list[float]], int | None]:
    """Embed ``texts`` one-at-a-time with BGE-M3 NaN-skip backfill.

    Logs the whole call as a single ledger row with the caller-chosen
    stage. Returns ``(vectors, embed_dim)``. Vectors of failed inputs
    are zero-filled to ``embed_dim`` so downstream cosine ranking
    treats them as unranked rather than length-mismatching.

    BGE-M3 served by Ollama occasionally produces NaN values for
    short / sparse / formatting-only inputs; the server then 500s the
    WHOLE batch because Go's JSON encoder rejects NaN. Per-input
    embedding plus zero-vector backfill keeps a single bad string
    from killing the document.
    """
    import httpx  # local import keeps the top-level dependency surface unchanged

    all_vectors: list[list[float]] = []
    embed_dim: int | None = None
    with ledger.log_call(
        architecture="graphrag",
        stage=stage,
        model=embedder.model,
        prompt="\n\n".join(texts),
        run_index=run_index,
    ) as rec:
        for text in texts:
            try:
                r = embedder.embed([text])
                vec = r.embeddings[0] if r.embeddings else []
                if vec:
                    embed_dim = embed_dim or len(vec)
                all_vectors.append(vec or [])
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 500:
                    all_vectors.append([])
                else:
                    raise
        if embed_dim is not None:
            all_vectors = [
                v if v else [0.0] * embed_dim
                for v in all_vectors
            ]
        rec.uncached_input_tokens = max(
            1, sum(len(t) for t in texts) // 4
        )
        rec.cached_input_tokens = 0
        rec.output_tokens = 0
        rec.response_hash = sha256_hex(
            "|".join(f"{v[0]:.6f}" for v in all_vectors if v)
        )
    return all_vectors, embed_dim


def _embed_entity_descriptions(
    entities: list[_Entity],
    *,
    embedder: OllamaEmbedder,
    ledger: CostLedger,
    run_index: int,
) -> tuple[list[list[float]], int | None]:
    """Embed every entity description once per paper.

    Logged as ``stage=preprocess`` because — like the per-chunk
    extraction and per-community summarisation — these embeddings are
    part of the index build and are reused across every question on
    this paper. Microsoft graphrag persists them to a vector store at
    indexing time; we keep them in-memory inside the cached
    ``_GraphRAGState``.
    """
    entity_texts = [
        f"{e.name}: {e.description[:_ENT_DESC_CHAR_CAP]}"
        for e in entities
    ]
    if not entity_texts:
        return [], None
    return _safe_embed_many(
        entity_texts,
        embedder=embedder, ledger=ledger,
        run_index=run_index, stage=Stage.PREPROCESS,
    )


def _local_search_build_context(
    *,
    query: str,
    g,
    entities: list[_Entity],
    relationships: list[_Relationship],
    communities: list[set[str]],
    reports: list[_CommunityReport],
    chunks: list[str],
    entity_vecs: list[list[float]],
    embed_dim: int | None,
    embedder: OllamaEmbedder,
    ledger: CostLedger,
    run_index: int,
    top_k_entities: int = _TOP_K_ENTITIES,
    top_k_relationships: int = _TOP_K_RELATIONSHIPS,
    max_context_tokens: int = _MAX_CONTEXT_TOKENS,
) -> tuple[str, list[str]]:
    """Build the local-search context for the answerer.

    Entity-description vectors are passed in precomputed (built once
    per paper during preprocessing). The only per-question embed call
    is the query itself, logged as ``stage=retrieval``.

    Returns ``(packed_context, evidence_sentences)``. The
    evidence_sentences list is a flat list of human-readable strings
    pulled into the context, used by the architecture-level
    ArchitectureResult for evidence reporting.
    """
    if not entities:
        return "(no entities extracted)", []

    # Query embedding only — entity embeddings are already in
    # ``entity_vecs`` from preprocessing.
    query_vectors, _q_dim = _safe_embed_many(
        [query], embedder=embedder, ledger=ledger,
        run_index=run_index, stage=Stage.RETRIEVAL,
    )
    if not query_vectors or not query_vectors[0]:
        # Pathological: query failed to embed. Bail out gracefully.
        return (
            "(local-search embedding failed: query produced NaN on the"
            " BGE-M3 server)",
            [],
        )
    query_vec = query_vectors[0]
    if embed_dim is None:
        # Pathological case: every entity produced NaN at build time.
        return (
            "(local-search embedding failed: every entity description"
            " produced NaN on the BGE-M3 server)",
            [],
        )

    # 2. Top-k entities by query-entity-description cosine.
    scores = [_cosine(query_vec, v) for v in entity_vecs]
    indexed = sorted(range(len(scores)), key=lambda i: -scores[i])
    selected_idxs = indexed[:top_k_entities]
    selected = [entities[i] for i in selected_idxs]
    selected_names = {e.name for e in selected}

    # 3. Communities ranked by entity-match count + community size.
    e2c = _entity_to_community(communities)
    community_match_count: dict[int, int] = defaultdict(int)
    for e in selected:
        for c_idx in e2c.get(e.name, []):
            community_match_count[c_idx] += 1
    reports_by_id = {r.community_id: r for r in reports}
    ranked_community_ids = sorted(
        community_match_count.keys(),
        key=lambda c: (-community_match_count[c],
                       -reports_by_id[c].rank if c in reports_by_id else 0),
    )

    # 4. Per-selected-entity relationships (top-k by weight, with
    # Microsoft-style relationship-strength as tiebreaker). The
    # primary ranking signal is unchanged from the prior
    # implementation (chunk co-occurrence count); strength only breaks
    # ties between edges seen in the same number of chunks. For a
    # typical 5-15-chunk QASPER paper most edges appear exactly once,
    # so the tiebreaker is what surfaces tight (strength 9-10) over
    # weak (strength 1-3) relationships in the top-k cut.
    rels_by_node: dict[str, list[tuple[str, int, float, str]]] = defaultdict(list)
    for u, v, data in g.edges(selected_names, data=True):
        if u in selected_names or v in selected_names:
            weight = data.get("weight", 1)
            strength = data.get("strength", 5.0)
            desc = data.get("description", "")
            rels_by_node[u].append((v, weight, strength, desc))
            rels_by_node[v].append((u, weight, strength, desc))
    selected_relationship_lines: list[str] = []
    for e in selected:
        node_rels = sorted(
            rels_by_node.get(e.name, []),
            key=lambda r: (-r[1], -r[2]),
        )[:top_k_relationships]
        for other, weight, _strength, desc in node_rels:
            line = f"- **{e.name} ↔ {other}** (weight {weight}): {desc}"
            if line not in selected_relationship_lines:
                selected_relationship_lines.append(line)

    # 5. Text units (chunks) the selected entities appear in. Rank by
    # number of selected entities present + number of selected
    # relationships co-occurring in the same chunk.
    chunk_score: dict[int, int] = defaultdict(int)
    for e in selected:
        for cid in e.text_unit_ids:
            chunk_score[cid] += 1
    for r in relationships:
        if (r.source in selected_names or r.target in selected_names):
            if r.text_unit_id >= 0:
                chunk_score[r.text_unit_id] += 1
    ranked_chunk_ids = sorted(chunk_score.keys(), key=lambda c: -chunk_score[c])

    # 6. Pack context within budget split:
    # community_prop / text_unit_prop / (1 - sum) for entities+relationships.
    community_budget = int(max_context_tokens * _COMMUNITY_PROP)
    text_unit_budget = int(max_context_tokens * _TEXT_UNIT_PROP)
    local_budget = max_context_tokens - community_budget - text_unit_budget

    community_items: list[tuple[str, int]] = []
    for c_idx in ranked_community_ids:
        if c_idx not in reports_by_id:
            continue
        text = reports_by_id[c_idx].text
        community_items.append((text, _approx_token_count(text)))
    community_packed = _pack_within_budget(community_items, community_budget)

    text_unit_items: list[tuple[str, int]] = []
    for cid in ranked_chunk_ids:
        if cid < 0 or cid >= len(chunks):
            continue
        text_unit_items.append((chunks[cid], _approx_token_count(chunks[cid])))
    text_unit_packed = _pack_within_budget(text_unit_items, text_unit_budget)

    entity_lines: list[tuple[str, int]] = []
    for e in selected:
        line = f"- **{e.name}** (type: {e.type}): {e.description}"
        entity_lines.append((line, _approx_token_count(line)))
    rel_items = [(line, _approx_token_count(line)) for line in selected_relationship_lines]
    local_items = entity_lines + rel_items
    local_packed = _pack_within_budget(local_items, local_budget)

    # 7. Stitch the context.
    parts: list[str] = []
    if community_packed:
        parts.append("# Community Reports\n\n" + "\n\n---\n\n".join(community_packed))
    if local_packed:
        # Split entity lines from relationship lines for readability.
        ent_section = [l for l in local_packed if any(l == el[0] for el in entity_lines)]
        rel_section = [l for l in local_packed if any(l == rl[0] for rl in rel_items)]
        if ent_section:
            parts.append("# Entities\n\n" + "\n".join(ent_section))
        if rel_section:
            parts.append("# Relationships\n\n" + "\n".join(rel_section))
    if text_unit_packed:
        parts.append("# Source Text\n\n" + "\n\n---\n\n".join(text_unit_packed))

    context = "\n\n".join(parts) if parts else "(no local-search context could be packed within the token budget)"

    # Evidence is the union of community reports and text units used.
    evidence = list(community_packed) + list(text_unit_packed)
    return context, evidence


# ──────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────

def run_graphrag(
    *,
    document: str,
    query: str,
    options: dict[str, str] | None,
    answerer: AnswererProvider,
    answerer_model: str,
    embedder: OllamaEmbedder,
    ledger: CostLedger,
    run_index: int = 0,
    max_tokens: int = 256,
    summary_model: str | None = None,
    summary_answerer: AnswererProvider | None = None,
    prompt_style: str = "pilot",
    cached_state: _GraphRAGState | None = None,
) -> ArchitectureResult:
    """Run the faithful GraphRAG local-search pipeline.

    Returns an ArchitectureResult with the predicted answer and the
    union of community reports + chunk text passed into the answerer
    context as evidence sentences.

    ``summary_answerer`` lets entity extraction + per-community
    summarisation run on a different provider than the final answer
    call (Phase F extension protocol: e.g. Grok final answer +
    Gemini Flash Lite preprocessing). When ``None`` the summary
    stage uses the same provider as the answerer.

    ``cached_state`` — when provided, skip steps 1–4 (chunking, entity
    extraction with gleaning, graph + Louvain, community reports) and
    reuse the prior call's artefacts. Only the per-query local-search
    embedding pass and the final answerer call land in the ledger on
    this path. This is the on-disk realisation of the
    ``C_off^struct / n`` amortisation in the cost model
    (project.tex § 3.4.1); without it, the build cost would be paid
    once per question instead of once per document and the Pareto
    comparison against flat full-context would be invalid.
    """
    summary_model = summary_model or answerer_model
    summary_answerer = summary_answerer or answerer

    if cached_state is not None:
        chunks = cached_state.chunks
        entities = cached_state.entities
        relationships = cached_state.relationships
        g = cached_state.g
        communities = cached_state.communities
        reports = cached_state.reports
        entity_vecs = cached_state.entity_vecs
        embed_dim = cached_state.embed_dim
        state = cached_state
    else:
        # 1. Chunk.
        chunker = SentenceBoundaryChunker(
            chunk_size_tokens=_CHUNK_TOKENS, overlap_tokens=_CHUNK_OVERLAP_TOKENS
        )
        chunks = [c.text for c in chunker.chunk(document)]
        if not chunks:
            return ArchitectureResult(
                architecture="graphrag",
                predicted_answer="",
                failed=True,
                failure_reason="document_produced_no_chunks",
            )

        # 2. Extract entities + relationships (with one gleaning pass).
        entities, relationships = _extract_entities_per_chunk(
            chunks,
            answerer=summary_answerer, answerer_model=summary_model,
            ledger=ledger, run_index=run_index,
        )
        if not entities:
            return ArchitectureResult(
                architecture="graphrag",
                predicted_answer="",
                failed=True,
                failure_reason="no_entities_extracted",
            )

        # 3. Build graph + communities.
        g, communities, _edge_tu = _build_graph_and_communities(
            entities, relationships,
        )

        # 4. Per-community structured reports.
        reports = _summarise_communities(
            g, communities,
            answerer=summary_answerer, answerer_model=summary_model,
            ledger=ledger, run_index=run_index,
        )

        # 4.5. Embed every entity description once. Logged as
        # ``stage=preprocess`` (part of the index build), reused
        # across every question on this paper.
        entity_vecs, embed_dim = _embed_entity_descriptions(
            entities,
            embedder=embedder, ledger=ledger, run_index=run_index,
        )

        state = _GraphRAGState(
            chunks=chunks,
            entities=entities,
            relationships=relationships,
            g=g,
            communities=communities,
            reports=reports,
            entity_vecs=entity_vecs,
            embed_dim=embed_dim,
        )

    # 5. Local search builds the packed context (per-query: embeds
    # the query and packs context against the cached entity vectors).
    context, evidence = _local_search_build_context(
        query=query,
        g=g, entities=entities, relationships=relationships,
        communities=communities, reports=reports, chunks=chunks,
        entity_vecs=entity_vecs, embed_dim=embed_dim,
        embedder=embedder, ledger=ledger, run_index=run_index,
    )

    # 6. Final answer call against the packed context. Same shared
    # prompt contract (and prompt_style) as the other architectures.
    prompt = _render_prompt(
        context=context, query=query, options=options, prompt_style=prompt_style
    )
    with ledger.log_call(
        architecture="graphrag",
        stage=Stage.GENERATE,
        model=answerer_model,
        prompt=prompt,
        run_index=run_index,
        temperature=0.0,
        max_tokens=max_tokens,
    ) as rec:
        result = answerer.call(
            prompt,
            model=answerer_model,
            max_tokens=max_tokens,
            temperature=0.0,
            cache_control=CacheControl.EPHEMERAL_5MIN,
        )
        rec.uncached_input_tokens = result.uncached_input_tokens
        rec.cached_input_tokens = result.cached_input_tokens
        rec.output_tokens = result.output_tokens
        rec.provider_request_id = result.provider_request_id
        rec.response_hash = sha256_hex(result.text or "")

    return ArchitectureResult(
        architecture="graphrag",
        predicted_answer=result.text,
        retrieved_evidence_sentences=evidence,
        prompt_token_count=result.uncached_input_tokens + result.cached_input_tokens,
        response_token_count=result.output_tokens,
        preprocessing_state=state,
    )
