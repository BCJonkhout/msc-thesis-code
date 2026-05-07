# GraphRAG implementation — entry-point notes

**Status as of 2026-05-07.** Microsoft `graphrag>=3.0.9` is installed
(see `pyproject.toml`). The implementation shim is the deferred
Step-3 architecture work; estimated ~1–2 days per the research brief.

## Recommended Approach (per research brief)

**Option (a): Microsoft `graphrag` v3.0.9 + custom
`LLMCompletion`/`LLMEmbedding` shim driven via `graphrag.api`,
NOT the CLI.**

## Investigated entry points

- `graphrag.api.index.build_index(config, ...)` — programmatic
  indexing. `config` is a `GraphRagConfig` (Pydantic).
- `graphrag.api.query.basic_search`, `local_search`, `global_search`,
  `drift_search` — query modes. Pilot should use **local_search** for
  QASPER + NovelQA (specific factual questions, not sensemaking).
- `graphrag.config.models.graph_rag_config.GraphRagConfig` — root
  config. Key fields:
  - `completion_models: dict[str, ModelConfig]`
  - `embedding_models: dict[str, ModelConfig]`
  - `chunking`, `extract_graph`, `cluster_graph`,
    `community_reports`, `local_search`, `cache`, `vector_store`
- `graphrag_llm.completion.LLMCompletion` (ABC) +
  `graphrag_llm.completion.register_completion(name, cls)` — register
  a custom completion provider. Constructor takes
  `model_id, model_config, tokenizer, metrics_store, cache_key_creator,
  metrics_processor, rate_limiter, retrier, cache, **kwargs`.
- `graphrag_llm.embedding.LLMEmbedding` (ABC) +
  `graphrag_llm.embedding.register_embedding(name, cls)` — same
  pattern for embeddings.

## Shim work plan (estimated ~1 day focused)

1. Subclass `LLMCompletion`. Implement `completion`,
   `completion_async`, `tokenizer` property, `metrics_store` property.
   Route every call through `pilot.providers.AnswererProvider` and
   write a `Stage.GENERATE` (for answer) or `Stage.PREPROCESS` (for
   entity extraction + community summaries) ledger row.
2. Subclass `LLMEmbedding`. Route through `pilot.encoders.OllamaEmbedder`
   and write a `Stage.RETRIEVAL` ledger row.
3. Register both at module import time:
   `register_completion("pilot-ledger", PilotLedgerCompletion)` and
   `register_embedding("pilot-ledger-embed", PilotLedgerEmbedding)`.
4. Build a `GraphRagConfig` programmatically with
   `model_provider="pilot-ledger"` for completion_models and
   `model_provider="pilot-ledger-embed"` for embedding_models.
5. **Critical config locks** (per research brief gotchas):
   - `chunks.size: 600` (paper default; package default differs)
   - `chunks.overlap: 100`
   - `extract_graph.max_gleanings: 1`
   - `cluster_graph.max_cluster_size: 10`
   - `cluster_graph.seed: <fixed>` (graspologic + igraph seeds for
     reproducibility)
   - `community_reports.max_input_length: 8000`
   - `community_reports.max_length: 2000`
   - `cache.type: "none"` (disable platform-side cache so our
     per-provider cache measurements stay honest)
   - All `models.*.temperature: 0.0`
6. Drive the index build with `graphrag.api.index.build_index(config)`,
   then run queries with `graphrag.api.query.local_search(config, ...)`.
7. Wrap as `pilot.architectures.graphrag.run_graphrag(...)` matching
   the signature of `run_flat`/`run_naive_rag`/`run_raptor`.
8. Wire into `step_3_dry_run._invoke_architecture`.

## Specific gotchas to honour

- **Mode choice.** Use `local_search`, not `global_search`. The paper's
  global search is for sensemaking (corpus-level themes); QASPER and
  NovelQA are entity-specific factual queries.
- **Leiden seed.** Set BOTH `cluster_graph.seed` and the underlying
  igraph seed; otherwise community boundaries shift between runs and
  break cached community-report reuse.
- **Truncation events.** Cap
  `summarize_descriptions.max_input_length` and log truncation events
  to the ledger — heavily-mentioned characters in long novels can
  accumulate hundreds of descriptions exceeding the model context.
- **Claims/covariates: off** by default in the package; keep off for
  Step 3 to match the paper's lower-cost configuration.
- **DRIFT search: skip.** Newer than the paper; including it would
  benchmark a 2024-blog method against three published ones.

## Verification

After the shim runs end-to-end on one QASPER paper:

- Every ledger row has `architecture="graphrag"` and a real
  `provider_request_id`.
- The `Stage.PREPROCESS` row count is roughly: chunks × 1
  (entity extraction) + entities × 1 (description summarization) +
  communities × 1 (community report) ≈ tens to a few hundred per
  paper.
- The `Stage.GENERATE` row count is exactly 1 per query (the
  local-search answer call).
- `provider.cache_control` lookups land cached prefixes on calls
  2..N as expected (Anthropic-style caching is provider-level, not
  graphrag-level — graphrag's own `cache.type=none` doesn't affect
  this).

## Files to create

- `src/pilot/architectures/graphrag.py` — the shim + run_graphrag.
- `tests/test_graphrag_shim.py` — mock the LLMCompletion/Embedding
  protocols and verify the ledger writes.
- Wire into `src/pilot/cli/step_3_dry_run.py:_invoke_architecture`.
- Update `pilot.architectures.__init__` to export `run_graphrag`.

## References

- Microsoft GraphRAG paper: arXiv:2404.16130 (Edge et al. 2024).
- Official repo: github.com/microsoft/graphrag.
- This pilot's research brief (in conversation transcript;
  reproduced as the inline notes above).
