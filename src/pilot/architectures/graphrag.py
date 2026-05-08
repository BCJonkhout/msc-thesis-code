"""GraphRAG runner — faithful from-scratch implementation.

Implements the GraphRAG algorithm of Edge et al. 2024 §3.1
(arXiv:2404.16130) against the pilot's existing primitives —
`OllamaEmbedder`, `AnswererProvider`, `CostLedger`,
`SentenceBoundaryChunker` — instead of pulling in Microsoft's
`graphrag` package and its complex `LLMCompletion` /
`LLMEmbedding` protocol layer. The from-scratch route is option
(c) from the research brief: it avoids the 10-class shim work
required to make the official package observable through the
pilot's cost ledger, at the cost of skipping the package's
hierarchical-Leiden / claims-extraction / DRIFT-search niceties
that aren't in the paper's headline algorithm anyway.

Pipeline (per Edge et al. §3.1 with paper-faithful parameters):

  1. Chunk source documents at 600 tokens / 100 overlap
     (the paper's choice; differs from naive_rag's 384/0).
  2. Per-chunk LLM call extracting entities (name, type,
     description) and relationships (source, target, description).
  3. Build a NetworkX graph: nodes = entities, edges = rels.
     Entities and edges merged by exact-string key across chunks;
     descriptions concatenated with newline separators (we skip
     the per-entity description-summarization LLM call here for
     budget reasons; flag in the paper's results table that this
     is a deliberate simplification of one of the paper's stages).
  4. Louvain community detection (NetworkX 3.x built-in).
  5. Per-community LLM call summarising the contained entities +
     edges into a "community report".
  6. Local search at query time: embed the query, top-k cosine
     against entity-name + community-report embeddings,
     concatenate the top reports into a context, single answerer
     call. Returns the answer + retrieved community-report text.

Cost-accounting notes
---------------------

Every LLM call writes a `CostLedger` row:

  - per-chunk entity extraction → ``Stage.PREPROCESS``
  - per-community summarisation → ``Stage.PREPROCESS``
  - query-time embedding (chunks + entities + reports + query) →
    ``Stage.RETRIEVAL``
  - final answer call → ``Stage.GENERATE``

The pilot's deployment-cost rule (Option A, run_index=0) applies
uniformly. T=0 for all calls per pilot plan § 3.4.3.

Deliberate simplifications vs the official Microsoft package
-------------------------------------------------------------

  - **No description-summary LLM call per entity / edge.** The
    paper's algorithm calls the LLM once per heavily-mentioned
    entity to summarise concatenated descriptions; we keep raw
    concatenated descriptions to avoid the per-entity cost on
    long documents (could be hundreds of LLM calls on a NovelQA
    novel). Pilot-paper-faithful enough; revisit at Step 5 if a
    sensitivity ablation shows it matters.
  - **Louvain instead of hierarchical Leiden.** The paper uses
    Leiden via graspologic; we use NetworkX's Louvain. Both are
    modularity-optimising; results differ in stability under
    multiple invocations (Leiden is provably more stable). Seed
    is fixed for reproducibility.
  - **No claims/covariates extraction.** The paper reports
    results both with and without; the package's default is
    without. We follow the without variant to keep cost down.
  - **Local search only.** QASPER + NovelQA are entity-specific
    factual-question workloads; the paper's global search is for
    sensemaking and would mismatch the task.

These simplifications are documented in the architecture note in
``thesis-msc/notes/pilot_findings.md`` so reviewers can see the
delta between this pilot's GraphRAG and the published method.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass

# Pin numba/OpenMP threads before networkx and friends import them.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from pilot.architectures.base import ArchitectureResult, _render_prompt
from pilot.encoders import OllamaEmbedder, SentenceBoundaryChunker
from pilot.ledger import CostLedger, Stage, sha256_hex
from pilot.providers.base import AnswererProvider, CacheControl


# ──────────────────────────────────────────────────────────────────────
# Paper-default parameters
# ──────────────────────────────────────────────────────────────────────

_CHUNK_TOKENS = 600   # Edge et al. §4.1.1
_CHUNK_OVERLAP_TOKENS = 100
_TOP_K_REPORTS = 3    # local-search: top-k community reports to pack into context
_TOP_K_ENTITIES = 8   # local-search: top-k entities by query embedding
_COMMUNITY_SEED = 0xDEADBEEF  # graspologic's default; preserved for reproducibility
_MAX_CONTEXT_TOKENS = 12_000  # paper §3.1.6 local-search default


# ──────────────────────────────────────────────────────────────────────
# Prompts (compact paper-faithful versions; full prompts at Step 5
# ablation if a wording change matters)
# ──────────────────────────────────────────────────────────────────────

_ENTITY_EXTRACT_PROMPT = """You are extracting structured information from a document chunk.

Identify all named entities (people, places, organisations, events) and the relationships between them. Respond with strict JSON of the form:

{{"entities": [{{"name": "...", "type": "...", "description": "..."}}, ...],
 "relationships": [{{"source": "...", "target": "...", "description": "..."}}, ...]}}

Use the entity name exactly as it appears in the text. Keep descriptions to one sentence. If a chunk has no entities, return {{"entities": [], "relationships": []}}.

Document chunk:
{chunk}
"""

_COMMUNITY_SUMMARY_PROMPT = """You are writing a brief report on a thematic community of entities. The community contains the following entities and their pairwise relationships.

Entities:
{entities}

Relationships:
{relationships}

Produce a 2-4 sentence summary covering: who/what is in the community, what they have in common, and the most important relationship in the community. Respond with the summary text only.
"""


@dataclass(frozen=True)
class _Entity:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class _Relationship:
    source: str
    target: str
    description: str


# ──────────────────────────────────────────────────────────────────────
# JSON-extraction helpers (defensive against LLMs that wrap output
# in markdown code fences)
# ──────────────────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def _parse_extract_json(raw: str) -> dict:
    """Lenient JSON extractor for the entity-extraction LLM output.

    Models occasionally wrap JSON in ```json ... ``` fences or emit
    explanatory prose around the JSON. Try strict json.loads first;
    fall back to the largest matching {...} block.
    """
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


# ──────────────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────────────

def _extract_entities_per_chunk(
    chunks: list[str],
    *,
    answerer: AnswererProvider,
    answerer_model: str,
    ledger: CostLedger,
    run_index: int,
    max_tokens: int = 1024,
) -> tuple[list[_Entity], list[_Relationship]]:
    """One LLM call per chunk extracting entities + relationships."""
    entities: dict[str, _Entity] = {}
    relationships: list[_Relationship] = []

    for chunk in chunks:
        prompt = _ENTITY_EXTRACT_PROMPT.format(chunk=chunk)
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
        for raw_e in parsed.get("entities", []) or []:
            if not isinstance(raw_e, dict):
                continue
            name = (raw_e.get("name") or "").strip()
            if not name:
                continue
            etype = (raw_e.get("type") or "").strip() or "unknown"
            desc = (raw_e.get("description") or "").strip()
            existing = entities.get(name)
            merged_desc = (
                f"{existing.description}\n{desc}" if existing and desc else (desc or (existing.description if existing else ""))
            )
            entities[name] = _Entity(name=name, type=existing.type if existing else etype, description=merged_desc.strip())

        for raw_r in parsed.get("relationships", []) or []:
            if not isinstance(raw_r, dict):
                continue
            src = (raw_r.get("source") or "").strip()
            tgt = (raw_r.get("target") or "").strip()
            if not src or not tgt:
                continue
            desc = (raw_r.get("description") or "").strip()
            relationships.append(_Relationship(source=src, target=tgt, description=desc))

    return list(entities.values()), relationships


def _build_graph_and_communities(
    entities: list[_Entity],
    relationships: list[_Relationship],
    *,
    seed: int = _COMMUNITY_SEED,
) -> tuple[object, list[set[str]]]:
    """Construct a NetworkX graph and run Louvain community detection."""
    import networkx as nx

    g = nx.Graph()
    for e in entities:
        g.add_node(e.name, type=e.type, description=e.description)
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    edge_descriptions: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in relationships:
        # Edge order normalised so (a, b) and (b, a) merge.
        key = tuple(sorted((r.source, r.target)))
        edge_counts[key] += 1
        if r.description:
            edge_descriptions[key].append(r.description)
        # Add nodes for entities first introduced in relationships.
        for n in key:
            if n not in g:
                g.add_node(n, type="unknown", description="")
    for key, count in edge_counts.items():
        g.add_edge(
            key[0], key[1],
            weight=count,
            description="\n".join(edge_descriptions[key]),
        )

    if g.number_of_nodes() == 0:
        return g, []

    # NetworkX 3.x's louvain_communities; seed for reproducibility.
    rng = random.Random(seed)
    communities = nx.community.louvain_communities(g, seed=rng.randint(0, 2**31 - 1))
    return g, communities


def _summarise_communities(
    g,
    communities: list[set[str]],
    *,
    answerer: AnswererProvider,
    answerer_model: str,
    ledger: CostLedger,
    run_index: int,
    max_tokens: int = 512,
) -> list[dict]:
    """Per-community LLM call producing a 2-4 sentence summary."""
    reports: list[dict] = []
    for community_idx, members in enumerate(communities):
        if not members:
            continue
        member_list = sorted(members)
        ent_lines = []
        for name in member_list:
            data = g.nodes[name]
            ent_lines.append(
                f"- {name} (type: {data.get('type', 'unknown')}): {data.get('description', '')[:200]}"
            )
        rel_lines = []
        for u, v, data in g.edges(member_list, data=True):
            if u in members and v in members:
                rel_lines.append(f"- {u} ↔ {v}: {data.get('description', '')[:200]}")

        prompt = _COMMUNITY_SUMMARY_PROMPT.format(
            entities="\n".join(ent_lines),
            relationships="\n".join(rel_lines) or "(none)",
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

        reports.append({
            "community_idx": community_idx,
            "members": member_list,
            "summary": result.text or "",
        })
    return reports


def _local_search(
    *,
    query: str,
    entities: list[_Entity],
    reports: list[dict],
    embedder: OllamaEmbedder,
    ledger: CostLedger,
    run_index: int,
    top_k_reports: int = _TOP_K_REPORTS,
) -> list[str]:
    """Embed the query + community-report summaries; return top-k summaries.

    The retrieval signal in our minimal GraphRAG is the cosine
    similarity between the query and each community report's
    summary embedding. Entity-name embeddings could also be used
    (the paper does both); we keep it to community reports for
    simplicity and cost.
    """
    if not reports:
        return []

    summaries = [r["summary"] for r in reports]
    embed_payload = "\n\n".join([query] + summaries)

    with ledger.log_call(
        architecture="graphrag",
        stage=Stage.RETRIEVAL,
        model=embedder.model,
        prompt=embed_payload,
        run_index=run_index,
    ) as rec:
        embed_result = embedder.embed([query] + summaries)
        rec.uncached_input_tokens = max(1, sum(len(t) for t in [query] + summaries) // 4)
        rec.cached_input_tokens = 0
        rec.output_tokens = 0
        rec.response_hash = sha256_hex(
            "|".join(f"{v[0]:.6f}" for v in embed_result.embeddings if v)
        )

    query_vec = embed_result.embeddings[0]
    report_vecs = embed_result.embeddings[1:]
    scores = [_cosine(query_vec, vec) for vec in report_vecs]
    indexed = sorted(range(len(scores)), key=lambda i: -scores[i])
    top = indexed[:top_k_reports]
    return [summaries[i] for i in top]


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
    """Run the GraphRAG pipeline (extract → graph → communities → local search → answer).

    Returns an ArchitectureResult with the predicted answer and the
    retrieved community-report summaries as evidence. The pipeline
    is single-pass and stateless across calls — every invocation
    re-runs entity extraction and community detection on the
    document, which is the same thing the cost model assumes for
    a deployment-cost calculation. Caching across queries on the
    same document would be the obvious follow-up optimisation.

    ``summary_answerer`` lets entity extraction + per-community
    summarisation run on a different provider than the final answer
    call (Phase F extension protocol: e.g. Grok final answer +
    Gemini Flash Lite preprocessing). When ``None`` the summary
    stage uses the same provider as the answerer.
    """
    summary_model = summary_model or answerer_model
    summary_answerer = summary_answerer or answerer

    # 1. Chunk the document at the paper-default 600 / 100.
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

    # 2. Extract entities + relationships per chunk.
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
    g, communities = _build_graph_and_communities(entities, relationships)

    # 4. Per-community summaries.
    reports = _summarise_communities(
        g, communities,
        answerer=summary_answerer, answerer_model=summary_model,
        ledger=ledger, run_index=run_index,
    )

    # 5. Local search: pick the top-k community-report summaries.
    top_summaries = _local_search(
        query=query, entities=entities, reports=reports,
        embedder=embedder, ledger=ledger, run_index=run_index,
    )

    # 6. Final answer call.
    context = "\n\n".join(top_summaries) if top_summaries else "(no community reports retrieved)"
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
        retrieved_evidence_sentences=top_summaries,
        prompt_token_count=result.uncached_input_tokens + result.cached_input_tokens,
        response_token_count=result.output_tokens,
    )
