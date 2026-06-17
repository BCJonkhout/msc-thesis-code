# Vendored: RAPTOR

This directory is a vendored copy of **RAPTOR** (Recursive Abstractive Processing
for Tree-Organized Retrieval), the tree-based hierarchical retrieval method by
Sarthi et al. It backs the `raptor` architecture in the benchmark
(`src/pilot/architectures/raptor.py` imports from here).

- **Upstream:** <https://github.com/parthsarthi03/raptor>
- **Paper:** Sarthi et al., *RAPTOR: Recursive Abstractive Processing for
  Tree-Organized Retrieval*, ICLR 2024 (`sarthi2024raptor` in the bibliography).
- **License:** MIT (see `LICENSE.txt`, © Parth Sarthi). Vendored unmodified except
  for the local patches enumerated below.

## Why it is vendored

RAPTOR's reference implementation hardcodes its own embedding, summarization, and
QA model classes. The benchmark needs every LLM/embedding call to (a) route
through the shared provider adapters and (b) write a cost-ledger row, and it needs
the clustering to be reproducible across runs. Vendoring lets the architecture
wrapper inject those without forking the algorithm. The tree-building and
retrieval logic is upstream's; only the seams below were changed.

## Local modifications (diff from upstream)

| File | Change | Why |
| --- | --- | --- |
| `cluster_utils.py` | `sys.setrecursionlimit(max(…, 10_000))` | deep recursive clustering on long novels overflows the default limit |
| `cluster_utils.py` | `RANDOM_SEED = 224`; seed Python `random`, NumPy, and the GMM `random_state` | reproducible cluster assignments across runs |
| `cluster_utils.py` | UMAP is deliberately **not** seeded (documented in-file) | cross-answerer determinism does not require it; see the inline comment for the rationale |
| `cluster_tree_builder.py` | summarization `ThreadPoolExecutor` bounded by `PILOT_BUILD_CONCURRENCY` (default 1) | cap concurrent LLM round-trips so the build is rate-limit-safe and the cost ledger stays attributable |
| `EmbeddingModels.py`, `QAModels.py` | upstream model classes left in place but bypassed | the architecture wrapper supplies the Ollama BGE-M3 embedder and the project answerer instead (see the in-file comments) |

The algorithm itself (recursive clustering, tree construction, collapsed-tree
retrieval) is unchanged from upstream. Do not edit the `.py` logic here for
cleanup reasons — this code is on the RAPTOR result-reproducibility path.
