# Long-Context QA Benchmark — Experiment Code

Companion code for the MSc thesis comparing four QA architectures on
repeated-context workloads under standardized cost accounting:

- **Flat full-context** (with cache-aware serving)
- **Naive RAG** (chunk-based retrieval)
- **RAPTOR** (tree-based hierarchical retrieval)
- **GraphRAG** (graph-based community summarization)

Datasets: QASPER, Core NovelQA (excluding B48), QuALITY.

The thesis paper lives in `../thesis-msc/`. The pilot setup plan that
drives this code is `../thesis-msc/notes/pilot_setup_plan.md`.
**Always read the pilot plan first** — it documents every locked
decision (sampling temperature `T = 0`, lit-anchored hyperparameters,
cost-attribution rule, etc.) and the rule that produced each one.

## Planned layout

```
code/
├── configs/          # YAML configs for models, methods, prices, embedding
│   ├── models.yaml
│   ├── methods.yaml         # lit-anchored hyperparameters with citations
│   ├── price_card.yaml      # rates and storage horizon
│   └── embedding.yaml
├── cost_ledger/      # per-call ledger writer; price-card resolver
├── prompts/          # qa_freeform.txt, qa_multiplechoice.txt
├── sanity/           # KV-cache verification harness, smoke runners
├── embedding/        # encoder wrapper + FAISS flat index
├── flat/             # flat full-context runner
├── naive_rag/        # chunk → embed → retrieve → answer
├── raptor/           # RAPTOR tree builder + collapsed-tree retrieval
├── graphrag/         # entity extraction + community summaries + retrieval
├── eval/             # Answer-F1, Evidence-F1, Accuracy, clustered bootstrap CI
├── pilot/            # pilot-step entrypoints (Step 0 .. Step 7)
├── data/             # local datasets (gitignored)
├── outputs/          # ledgers, embeddings, trees, graphs (gitignored)
└── env/              # conda lock + container definitions
```

## Reproducibility provenance rule

Every locked value in `configs/*.yaml` carries a `source:` field that
points at one of:

- A pilot decision-matrix row reference (`pilot:5.8#12` for the `N`
  decision, etc.).
- A literature citation key (`lit:sarthi2024raptor`, `lit:edge2024local`).
- A methodology rule reference (`rule:option-A`).

No locked value is allowed without a `source:` field.

## Status

Empty scaffold as of repo init. Implementation begins with Step 0
(cost ledger, prompt template harness, KV-cache verification) per the
pilot plan procedure.
