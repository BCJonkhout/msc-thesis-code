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

_ENTITY_EXTRACT_PROMPT = """You are extracting structured information from a document chunk.

Identify all named entities (people, places, organisations, events, concepts, methods, datasets) and the relationships between them. Respond with strict JSON of the form:

{{"entities": [{{"name": "...", "type": "...", "description": "..."}}, ...],
 "relationships": [{{"source": "...", "target": "...", "description": "..."}}, ...]}}

Use the entity name exactly as it appears in the text. Keep descriptions to one sentence. If a chunk has no entities, return {{"entities": [], "relationships": []}}.

Document chunk:
{chunk}
"""

# Microsoft graphrag's gleaning loop appends this single prompt and
# expects the model to emit additional entities/relationships missed
# in the first pass. The format is the same as the initial extract.
_ENTITY_GLEAN_PROMPT = """You previously extracted entities and relationships from this document chunk. Some entities or relationships may have been missed.

Re-examine the chunk and emit any ADDITIONAL entities and relationships that were not in your prior response. Use the same JSON format. If you found nothing new, return {{"entities": [], "relationships": []}}.

Document chunk:
{chunk}
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
    # Per-edge weight is the count of distinct chunks the (source, target)
    # pair appeared in; computed at graph-build time.


@dataclass
class _CommunityReport:
    community_id: int
    member_names: list[str]
    text: str          # rendered markdown report
    rank: int          # community size (proxy for Microsoft's rank attribute)


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
        relationships.append(_Relationship(
            source=src, target=tgt, description=desc, text_unit_id=chunk_idx,
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

    for chunk_idx, chunk in enumerate(chunks):
        for pass_idx in range(max_gleanings + 1):
            if pass_idx == 0:
                prompt = _ENTITY_EXTRACT_PROMPT.format(chunk=chunk)
            else:
                prompt = _ENTITY_GLEAN_PROMPT.format(chunk=chunk)
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

            parsed = _parse_extract_json(result.text or "")
            # Stop gleaning early if a pass returns nothing
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
    for r in relationships:
        key = tuple(sorted((r.source, r.target)))
        edge_counts[key] += 1
        if r.description:
            edge_descriptions[key].append(r.description)
        edge_text_units[key].append(r.text_unit_id)
        for n in key:
            if n not in g:
                g.add_node(n, type="unknown", description="", text_unit_ids=())
    for key, count in edge_counts.items():
        g.add_edge(
            key[0], key[1],
            weight=count,
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
    """Per-community LLM call producing a structured ~500-700 token report."""
    reports: list[_CommunityReport] = []
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

        reports.append(_CommunityReport(
            community_id=community_idx,
            member_names=member_list,
            text=result.text or "",
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


def _local_search_build_context(
    *,
    query: str,
    g,
    entities: list[_Entity],
    relationships: list[_Relationship],
    communities: list[set[str]],
    reports: list[_CommunityReport],
    chunks: list[str],
    embedder: OllamaEmbedder,
    ledger: CostLedger,
    run_index: int,
    top_k_entities: int = _TOP_K_ENTITIES,
    top_k_relationships: int = _TOP_K_RELATIONSHIPS,
    max_context_tokens: int = _MAX_CONTEXT_TOKENS,
) -> tuple[str, list[str]]:
    """Build the local-search context for the answerer.

    Returns ``(packed_context, evidence_sentences)``. The
    evidence_sentences list is a flat list of human-readable strings
    pulled into the context, used by the architecture-level
    ArchitectureResult for evidence reporting.
    """
    if not entities:
        return "(no entities extracted)", []

    # 1. Embed query + every entity description.
    #
    # Heavily-mentioned entities accumulate concatenated descriptions
    # across every chunk they appear in; for a long document with
    # many chunks an entity description can grow to several thousand
    # characters. Truncate each entity-text to a generous prefix —
    # the embedding signal is preserved without sending the entire
    # multi-thousand-character concatenation through Ollama.
    _ENT_DESC_CHAR_CAP = 1000
    entity_texts = [
        f"{e.name}: {e.description[:_ENT_DESC_CHAR_CAP]}"
        for e in entities
    ]
    embed_payload = [query] + entity_texts

    # BGE-M3 served by Ollama occasionally produces NaN values in
    # embedding vectors for short / sparse / formatting-only inputs;
    # the server then returns 500 on the WHOLE batch because Go's
    # JSON encoder rejects NaN. Embed each input individually so a
    # single bad entity doesn't kill the document. Skip entries that
    # 500 by substituting a zero vector — they sort to the bottom of
    # cosine ranking and are effectively excluded from local-search
    # retrieval without aborting the whole architecture run.
    import httpx  # local import keeps the top-level dependency surface unchanged
    all_vectors: list[list[float]] = []
    skipped = 0
    embed_dim: int | None = None
    with ledger.log_call(
        architecture="graphrag",
        stage=Stage.RETRIEVAL,
        model=embedder.model,
        prompt="\n\n".join(embed_payload),
        run_index=run_index,
    ) as rec:
        for text in embed_payload:
            try:
                r = embedder.embed([text])
                vec = r.embeddings[0] if r.embeddings else []
                if vec:
                    embed_dim = embed_dim or len(vec)
                all_vectors.append(vec or [])
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 500:
                    skipped += 1
                    all_vectors.append([])  # dim resolved on first success
                else:
                    raise
        # Backfill zero vectors for the BGE-M3 NaN skips so the
        # downstream cosine sort treats them as unranked rather than
        # crashing on length mismatch.
        if embed_dim is not None:
            all_vectors = [
                v if v else [0.0] * embed_dim
                for v in all_vectors
            ]
        elif all_vectors:
            # Pathological case: every input produced NaN. Bail out
            # gracefully rather than computing similarity on empty
            # vectors.
            return (
                "(local-search embedding failed: every entity description"
                " produced NaN on the BGE-M3 server)",
                [],
            )
        rec.uncached_input_tokens = max(
            1, sum(len(t) for t in embed_payload) // 4
        )
        rec.cached_input_tokens = 0
        rec.output_tokens = 0
        rec.response_hash = sha256_hex(
            "|".join(
                f"{v[0]:.6f}" for v in all_vectors if v
            )
        )
    query_vec = all_vectors[0]
    entity_vecs = all_vectors[1:]

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

    # 4. Per-selected-entity relationships (top-k by weight).
    rels_by_node: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for u, v, data in g.edges(selected_names, data=True):
        if u in selected_names or v in selected_names:
            rels_by_node[u].append((v, data.get("weight", 1), data.get("description", "")))
            rels_by_node[v].append((u, data.get("weight", 1), data.get("description", "")))
    selected_relationship_lines: list[str] = []
    for e in selected:
        node_rels = sorted(rels_by_node.get(e.name, []), key=lambda r: -r[1])[:top_k_relationships]
        for other, weight, desc in node_rels:
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
    """
    summary_model = summary_model or answerer_model
    summary_answerer = summary_answerer or answerer

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

    # 5. Local search builds the packed context.
    context, evidence = _local_search_build_context(
        query=query,
        g=g, entities=entities, relationships=relationships,
        communities=communities, reports=reports, chunks=chunks,
        embedder=embedder, ledger=ledger, run_index=run_index,
    )

    # 6. Final answer call against the packed context.
    prompt = _render_prompt(context=context, query=query, options=options)
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
    )
