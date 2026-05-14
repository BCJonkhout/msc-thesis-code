"""Naive RAG runner.

The "naive RAG" baseline as it appears in modern long-document QA
literature (e.g. RAPTOR §4 baselines, GraphRAG §4.1 baselines): no
query rewriting, no reranking, no recursive control. Just:

  1. (Build, once per paper) Chunk the document at the size in
     ``methods.yaml#naive_rag`` (sentence-boundary-aware; the pilot
     uses 384 tokens / 0 overlap) and embed every chunk via the
     configured encoder (BGE-M3 served by Ollama). The chunk index
     is reused across every question on this document.
  2. (Query, once per question) Embed the question and cosine-rank
     against the cached chunk vectors; take top-k.
  3. Concatenate the top-k chunk texts into the prompt's ``{context}``
     slot and call the answerer once.

The retrieved chunk texts are returned as
``retrieved_evidence_sentences`` so QASPER's Evidence-F1 diagnostic
can compare them against gold sentences.

Cost-accounting note: chunk-embedding is logged as
``Stage.PREPROCESS`` because it is paid once per (paper, arch) at
build time — same amortisation rule as RAPTOR's tree-build embeds
and GraphRAG's entity-description embeds. The per-question query
embedding lands as ``Stage.RETRIEVAL``; the answerer call as
``Stage.GENERATE``. On a cache hit (Q2..QN), only the latter two
rows are emitted.

An earlier version of this runner re-chunked and re-embedded the
entire document on every question, which logged the build cost as
``Stage.RETRIEVAL`` and multiplied it by the number of questions
per document. For NovelQA's 90k-token novels at 384-token chunks
this meant ~250 redundant BGE-M3 calls per question and a ~5x
inflation of Naive RAG's measured per-question cost — a systematic
bias against Naive RAG in the Pareto comparison against RAPTOR /
GraphRAG (whose build cost was already correctly amortised). The
build/query split here closes that gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from pilot.architectures.base import ArchitectureResult, _render_prompt
from pilot.encoders import OllamaEmbedder, SentenceBoundaryChunker
from pilot.ledger import CostLedger, Stage, sha256_hex
from pilot.providers.base import AnswererProvider, CacheControl


@dataclass
class _NaiveRagState:
    """Per-(run, paper) preprocessing artefact for Naive RAG.

    Carries the chunked document text and the embedding for every
    chunk so subsequent questions on the same paper only pay one
    query embed + cosine ranking + one answerer call.
    """
    chunk_texts: list[str]
    chunk_vecs: list[list[float]] = field(default_factory=list)


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def _topk_indices(scores: list[float], k: int) -> list[int]:
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: (-x[1], x[0]))
    return [i for i, _ in indexed[:k]]


def run_naive_rag(
    *,
    document: str,
    query: str,
    options: dict[str, str] | None,
    answerer: AnswererProvider,
    answerer_model: str,
    embedder: OllamaEmbedder,
    chunker: SentenceBoundaryChunker,
    ledger: CostLedger,
    top_k: int = 8,
    run_index: int = 0,
    cache_control: CacheControl = CacheControl.EPHEMERAL_5MIN,
    max_tokens: int = 256,
    temperature: float = 0.0,
    prompt_style: str = "pilot",
    cached_state: _NaiveRagState | None = None,
) -> ArchitectureResult:
    """Naive RAG: chunk + embed (build) → query embed + top-k cosine → answer.

    ``cached_state`` — when provided, skip chunking and chunk-embedding
    (the build-time work) and reuse the cached chunk vectors. The
    cost ledger therefore records the chunk-index build exactly once
    per (run, paper_id, arch); subsequent questions on that paper
    pay only one query-embed row and one answer-generate row.
    """
    if cached_state is not None:
        chunk_texts = cached_state.chunk_texts
        chunk_vecs = cached_state.chunk_vecs
        state = cached_state
    else:
        chunks = chunker.chunk(document)
        if not chunks:
            return ArchitectureResult(
                architecture="naive_rag",
                predicted_answer="",
                retrieved_evidence_sentences=[],
                failed=True,
                failure_reason="document_produced_no_chunks",
            )

        # Build phase: chunk-embed the entire document once. Token
        # count is approximated by the same char/4 heuristic used by
        # the other architectures' embed paths (Ollama doesn't report
        # token usage on /api/embed). Logged as ``Stage.PREPROCESS``
        # because the chunk index is a build-time artefact reused
        # across every question on this paper.
        chunk_texts = [c.text for c in chunks]
        chunk_embed_payload = "\n\n".join(chunk_texts)
        estimated_chunk_tokens = max(1, sum(len(t) for t in chunk_texts) // 4)
        with ledger.log_call(
            architecture="naive_rag",
            stage=Stage.PREPROCESS,
            model=embedder.model,
            prompt=chunk_embed_payload,
            run_index=run_index,
            temperature=temperature,
        ) as rec:
            chunk_embed_result = embedder.embed(chunk_texts)
            rec.uncached_input_tokens = estimated_chunk_tokens
            rec.cached_input_tokens = 0
            rec.output_tokens = 0
            rec.response_hash = sha256_hex(
                "|".join(
                    f"{v[0]:.6f}" for v in chunk_embed_result.embeddings if v
                )
            )
        chunk_vecs = chunk_embed_result.embeddings
        state = _NaiveRagState(chunk_texts=chunk_texts, chunk_vecs=chunk_vecs)

    # Query embedding — paid every question, logged as
    # ``Stage.RETRIEVAL``. One ledger row per question.
    estimated_query_tokens = max(1, len(query) // 4)
    with ledger.log_call(
        architecture="naive_rag",
        stage=Stage.RETRIEVAL,
        model=embedder.model,
        prompt=query,
        run_index=run_index,
        temperature=temperature,
    ) as rec:
        query_embed_result = embedder.embed([query])
        rec.uncached_input_tokens = estimated_query_tokens
        rec.cached_input_tokens = 0
        rec.output_tokens = 0
        rec.response_hash = sha256_hex(
            "|".join(
                f"{v[0]:.6f}" for v in query_embed_result.embeddings if v
            )
        )
    question_vec = query_embed_result.embeddings[0]

    scores = [_cosine(question_vec, vec) for vec in chunk_vecs]
    topk = _topk_indices(scores, top_k)

    retrieved_chunk_texts = [chunk_texts[i] for i in topk]
    context = "\n\n".join(retrieved_chunk_texts)
    prompt = _render_prompt(
        context=context, query=query, options=options, prompt_style=prompt_style
    )

    with ledger.log_call(
        architecture="naive_rag",
        stage=Stage.GENERATE,
        model=answerer_model,
        prompt=prompt,
        run_index=run_index,
        temperature=temperature,
        max_tokens=max_tokens,
    ) as rec:
        result = answerer.call(
            prompt,
            model=answerer_model,
            max_tokens=max_tokens,
            temperature=temperature,
            cache_control=cache_control,
        )
        rec.uncached_input_tokens = result.uncached_input_tokens
        rec.cached_input_tokens = result.cached_input_tokens
        rec.output_tokens = result.output_tokens
        rec.provider_request_id = result.provider_request_id
        rec.provider_region = result.provider_region
        rec.response_hash = sha256_hex(result.text or "")

    return ArchitectureResult(
        architecture="naive_rag",
        predicted_answer=result.text,
        retrieved_evidence_sentences=retrieved_chunk_texts,
        prompt_token_count=result.uncached_input_tokens + result.cached_input_tokens,
        response_token_count=result.output_tokens,
        preprocessing_state=state,
    )
