"""Naive RAG runner.

The "naive RAG" baseline as it appears in modern long-document QA
literature (e.g. RAPTOR §4 baselines, GraphRAG §4.1 baselines): no
query rewriting, no reranking, no recursive control. Just:

  1. Chunk the document at the size in `methods.yaml#naive_rag`
     (sentence-boundary-aware via langchain-text-splitters; the
     pilot uses 384 tokens / 0 overlap).
  2. Embed every chunk + the question via the configured encoder
     (BGE-M3 served by Ollama, locked at Step 3 encoder Recall@k).
  3. Cosine-rank chunks against the question; take top-k.
  4. Concatenate the top-k chunk texts into the prompt's
     `{context}` slot.
  5. Call the answerer once.

The retrieved chunk texts are returned as
`retrieved_evidence_sentences` so QASPER's Evidence-F1 diagnostic
can compare them against gold sentences.

Cost-accounting note: the embedding stage writes one ledger row at
Stage.RETRIEVAL with output_tokens=0 (embedders don't have an output
in the LLM sense; the cost model treats embed input tokens as the
billable quantity). The answerer call writes one Stage.GENERATE row.
So a single Naive RAG (architecture, query) pair produces exactly
two ledger rows, regardless of how many chunks the document is split
into.
"""
from __future__ import annotations

import math

from pilot.architectures.base import ArchitectureResult, _render_prompt
from pilot.encoders import OllamaEmbedder, SentenceBoundaryChunker
from pilot.ledger import CostLedger, Stage, sha256_hex
from pilot.providers.base import AnswererProvider, CacheControl


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
) -> ArchitectureResult:
    """Naive RAG: chunk → embed → top-k cosine retrieve → answer."""
    chunks = chunker.chunk(document)
    if not chunks:
        return ArchitectureResult(
            architecture="naive_rag",
            predicted_answer="",
            retrieved_evidence_sentences=[],
            failed=True,
            failure_reason="document_produced_no_chunks",
        )

    # Embed chunks + question. The embed call is logged once with the
    # combined input as the prompt-hash payload; embedders return no
    # output text, so output_tokens stays at 0. The token count is
    # measured by the embedder's own tokenizer at the provider side
    # — Ollama doesn't expose token usage on /api/embed, so we
    # estimate by character count (4 chars/token heuristic). This
    # under-reports slightly but is the same heuristic across all
    # architectures so the comparison is fair.
    chunk_texts = [c.text for c in chunks]
    embed_inputs = chunk_texts + [query]
    embed_payload = "\n\n".join(embed_inputs)
    estimated_input_tokens = max(1, sum(len(t) for t in embed_inputs) // 4)

    with ledger.log_call(
        architecture="naive_rag",
        stage=Stage.RETRIEVAL,
        model=embedder.model,
        prompt=embed_payload,
        run_index=run_index,
        temperature=temperature,
    ) as rec:
        embed_result = embedder.embed(embed_inputs)
        rec.uncached_input_tokens = estimated_input_tokens
        rec.cached_input_tokens = 0
        rec.output_tokens = 0
        rec.response_hash = sha256_hex(
            "|".join(f"{v[0]:.6f}" for v in embed_result.embeddings if v)
        )

    chunk_vecs = embed_result.embeddings[: len(chunks)]
    question_vec = embed_result.embeddings[-1]
    scores = [_cosine(question_vec, vec) for vec in chunk_vecs]
    topk = _topk_indices(scores, top_k)

    retrieved_chunk_texts = [chunks[i].text for i in topk]
    context = "\n\n".join(retrieved_chunk_texts)
    prompt = _render_prompt(context=context, query=query, options=options)

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
    )
