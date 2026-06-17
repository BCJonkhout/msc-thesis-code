# Long-Context QA Benchmark — Experiment Code

Companion code for the MSc thesis comparing four QA architectures on
long-document, repeated-context workloads under a single cost-accounting
framework:

- **Flat full-context** (cache-aware serving)
- **Naive RAG** (chunk-based retrieval)
- **RAPTOR** (tree-based hierarchical retrieval)
- **GraphRAG** (graph-based community summarization)

**Datasets:** **QASPER** (local Answer-F1) and **NovelQA** (held-out Codabench
multiple-choice gold).

The thesis paper lives in a sibling repository,
[`BCJonkhout/thesis-msc-paper`](https://github.com/BCJonkhout/thesis-msc-paper).
No metric, table, or figure is hardcoded in the paper: each is produced here and
imported. The map from producer script to paper asset is
**[`docs/CODEMAP.md`](docs/CODEMAP.md)** — read that to see how any number in the
paper was computed.

## Headline result

With a single answerer (`gemini-3.1-flash-lite-preview`, greedy decoding) and a
shared BGE-M3 encoder, **flat full-context wins both datasets**. The ranking is
identical across QASPER and NovelQA — flat > naive_rag ≈ raptor > graphrag — and
RAPTOR and GraphRAG sit *off* the cost–quality Pareto frontier. In the
long-context-model regime, added retrieval structure does not improve quality;
the live trade-off is flat (quality) vs. naive RAG (budget).

> **History.** The package is named `pilot/` for historical reasons. The study
> began as a calibration pilot whose job was to lock N, the answerer, the
> encoder, and the prompts. In doing so it surfaced and fixed three confounds
> (architecture prompt mis-routing, abstention-template fallthrough, and a
> consensus-oracle NovelQA proxy), then was re-run at scale against held-out
> gold. The pilot is retained as a methodology-validation step;
> [`docs/CODEMAP.md`](docs/CODEMAP.md) separates the canonical main-study scripts
> from the pilot-era reproducibility record.

---

## Layout

```text
code/
├── configs/                       # YAML configs (every value carries source:)
│   ├── models.yaml                # answerer slate + rejected_candidates ledger
│   ├── methods.yaml               # lit-anchored hyperparameters with citations
│   ├── price_card.yaml            # provider rates + storage horizon (base card)
│   ├── price_card_cache_discount.yaml  # second pre-registered cost card
│   └── embedding.yaml             # BGE-M3 default + escalation chain
├── docs/
│   ├── CODEMAP.md                 # producer-script → paper-asset map (start here)
│   └── graphrag_design_notes.md   # rejected-alternative design record
├── src/pilot/                     # the reusable library (see docs/CODEMAP.md)
│   ├── architectures/             # run_flat / run_naive_rag / run_raptor / run_graphrag
│   ├── providers/                 # AnswererProvider ABC + adapters + factory
│   ├── encoders/                  # BGE-M3 Ollama embedder + chunker
│   ├── codabench/                 # NovelQA submission + score recovery
│   ├── eval/metrics.py            # Answer-F1 and MC scoring
│   ├── cli/                       # pilot step harnesses (step_0 … step_4, phase_f)
│   ├── ledger.py, price_card.py   # append-only cost ledger + USD computation
│   └── provenance.py              # the source: gate enforced by the test suite
├── scripts/                       # analysis, figure, table, and launcher scripts
│   └── figures/                   # retired pilot figure renderers
├── third_party/raptor/            # vendored RAPTOR (see its README)
├── tests/                         # 357 passing tests (303 functions, 32 files); no live API calls
├── data/                          # local datasets (gitignored)
└── outputs/                       # ledgers, scored cells, results (gitignored)
```

---

## Reproducibility quickstart

The pipeline is deterministic given the same configs, seeds, and slate.

```bash
# 1. Prerequisites: Python 3.11+ (3.12 verified on Windows), uv, and Ollama
#    (local embedding inference for BGE-M3).

# 2. Sync dependencies (creates .venv/, generates uv.lock)
git clone https://github.com/BCJonkhout/msc-thesis-code.git
cd msc-thesis-code
uv sync --extra test

# 3. Credentials
cp .env.example .env        # fill in the providers you will use

# 4. Pull the embedding model
ollama pull bge-m3

# 5. Tests (fast, no API calls)
make test

# 6. Acquire data and run the study (see the Makefile for all targets)
make data-download          # QASPER + NovelQA → data/
make build-calibration      # deterministic calibration pool, seed=42
```

Results land under `outputs/` (gitignored). The producer scripts that turn a run
into the paper's tables and figures are documented in
[`docs/CODEMAP.md`](docs/CODEMAP.md); `make export-assets` promotes the finished
assets into `thesis-msc/generated/`.

---

## Result artifacts are sacred

Everything under `outputs/` and `data/` is a produced result or an input dataset.
It is never deleted or hand-edited and is excluded from version control, so it
lives only on the producing machine. Every number in the paper traces back to
these files; the cleanup convention is that any script which produced one of them
is kept (and documented in the CODEMAP) even when a later variant supersedes it.

---

## Provenance rule (every locked value)

Every value in `configs/*.yaml` carries a `source:` field pointing at one of:

| Source kind | Example | Meaning |
| --- | --- | --- |
| Decision-matrix row | `pilot:5.8#12` | a locked decision per the pilot plan |
| Literature citation key | `lit:sarthi2024raptor` | a citation in the thesis bibliography |
| Methodology rule | `rule:option-A` | a methodology rule |
| Provider documentation | `provider:platform.claude.com/docs/...` | pinned to a docs URL |
| Empirical measurement | `empirical:step_2_kvcache_2026-05-02` | a measurement made during the study |

`tests/test_provenance.py` fails the suite if any leaf config value lacks a
non-empty `source:`.

---

## Caching: provider adapter quirks

KV-cache verification (pilot Step 2) uncovered three adapter requirements that
would otherwise silently destroy a cache-amortization measurement. They are
retained in the adapters and matter to anyone reproducing the cost accounting:

1. **OpenRouter requires per-model upstream pinning.** Sticky routing only
   activates after a request is observed to use caching, so the adapter pins
   `provider.only=["deepseek"]` for `deepseek/*` slugs; slugs whose upstreams do
   not cache are routed unpinned to preserve availability.
2. **xAI requires the `x-grok-conv-id` header.** The cache is per-server; the
   adapter generates a per-instance UUID and sends it on every call so
   consecutive calls hit the same cache-warm server.
3. **Anthropic Opus 4.7 rejects `temperature=0`.** The adapter omits
   `temperature`/`top_p` for that model id and relies on model defaults.

The main study itself uses a single Gemini answerer; the multi-provider slate and
these quirks belong to the pilot's qualification and the cross-vendor probe.

---

## Datasets

All datasets land under `data/` (gitignored) and are not redistributed here.

### QASPER

Acquired from Allen AI's S3 release tarballs (the HuggingFace `allenai/qasper`
script-loader breaks on `datasets` v3+). Storage:
`data/qasper/{train,dev,test,calibration_pool}.jsonl`. The calibration pool is
drawn from dev with a "requires ≥1 annotated evidence sentence" filter so
Evidence-F1 is computable.

### NovelQA

Acquired from the gated `NovelQA/NovelQA` HuggingFace dataset via
`huggingface_hub.snapshot_download` + zip extraction (it is a flat file
collection, so `load_dataset` does not apply). To get access, click "Agree and
access" on the dataset card with the account whose token is in
`HUGGINGFACE_ACCESS_TOKEN`. Only the public-domain subset is evaluated
(copyright-withheld novels lack texts). **B48** (*The History of Rome*, 2.58M
tokens) is excluded from calibration sampling and from the main evaluation pool;
calibration novels B01, B08, B41, B50 are also held out of the test pool.
Storage: `data/novelqa/{full_texts/, questions.jsonl, calibration_pool.jsonl,
calibration_novels.json, bookmeta.json}`. Scored against held-out Codabench gold.

---

## Sampling temperature

`T = 0` (greedy decoding) is fixed across every LLM call in every architecture —
RAPTOR summarization, GraphRAG entity extraction + community summaries, and the
final answerer. The justification is in the paper (§ Sampling Temperature):
literature consensus on greedy decoding for QA evaluation, the falsifiability
requirement to hold sampling temperature constant across architectures, and
suppression of hallucinated labels in the preprocessing index. Greedy decoding is
not bit-for-bit deterministic end-to-end (floating-point non-associativity, batch
invariance, load balancing); the N=5 multi-run protocol absorbs the residual
variance, which at T=0 is near-zero on these tasks.

---

## Tests

```bash
make test          # or: .venv/Scripts/python.exe -m pytest tests/ -q
```

357 tests pass (303 test functions across 32 files); no test makes a live API
call. The suite
covers the cost ledger and USD computation, the prompt loader, the provenance
gate, MC post-processing, the provider factory, the architecture stages, the
resumable build / preprocess cache, crash-safety, the Codabench format/score
path, and the pilot orchestrators.

---

## Outputs and ledgers

- `outputs/main_study/` — the completed study: `scored_cells.jsonl` and the
  analysis JSONs (`significance.json`, `cost_per_arch.json`, `breakeven.json`,
  `memorization_control.json`); `export/` holds the paper-named assets.
- `outputs/runs/<run_id>/ledger.jsonl` — per-call cost ledger rows
  (`provider_request_id`, prompt/response hashes, wall-clock, token counts,
  seed, temperature, region).
- `outputs/sanity/` — per-step verdict files from the pilot harnesses.

All gitignored; recoverable by re-running the corresponding targets against the
same configs and slate.

---

## License and data terms

- Code: MIT (see `pyproject.toml`).
- Datasets retain their own licenses (QASPER per Allen AI; NovelQA Apache-2.0 plus
  the HF dataset-card agreement). No dataset content is redistributed here.
- `outputs/` and `data/` are excluded from version control to keep API-derived
  material and gated dataset content off the public repo.
