# Long-Context QA Benchmark — Experiment Code

Companion code for the MSc thesis comparing four QA architectures on
repeated-context workloads under standardized cost accounting:

- **Flat full-context** (with cache-aware serving)
- **Naive RAG** (chunk-based retrieval)
- **RAPTOR** (tree-based hierarchical retrieval)
- **GraphRAG** (graph-based community summarization)

Datasets: **QASPER**, **Core NovelQA** (excluding B48), **QuALITY**.

The thesis paper itself lives in a sibling repository:
[`BCJonkhout/thesis-msc-paper`](https://github.com/BCJonkhout/thesis-msc-paper).
That repository carries the LaTeX source plus the methodology notes
that drive this code:

- `notes/pilot_setup_plan.md` — every locked pilot decision
  (sampling temperature `T = 0`, lit-anchored hyperparameters,
  cost-attribution rule, etc.) and the rule that produced each.
  **Read this before changing anything in this code repo.**
- `notes/pilot_findings.md` — empirical findings per pilot step.
- `notes/cache_ttl_per_provider.md` — provider cache-TTL audit.

---

## Reproducibility quickstart

The pipeline is fully deterministic given the same configs, seeds,
and slate. To reproduce results from a clean clone:

```bash
# 1. Prerequisites
#    - Python 3.11+ (3.12 verified on Windows)
#    - uv (https://docs.astral.sh/uv/) for dependency resolution
#    - Ollama (https://ollama.com) for local embedding inference
#      (needed by the encoder Recall@k sub-experiment in Step 3)

# 2. Sync dependencies (creates .venv/, generates uv.lock)
git clone https://github.com/BCJonkhout/msc-thesis-code.git
cd msc-thesis-code
uv sync --extra test

# 3. Provide credentials
cp .env.example .env
# Edit .env and fill in the keys for whichever providers you'll use.
# At minimum you need: OPENROUTER_API_KEY, GEMINI_API_KEY (or
# GOOGLE_API_KEY), XAI_API_KEY, HUGGINGFACE_ACCESS_TOKEN.
# Optional: ANTHROPIC_API_KEY and OPENAI_API_KEY are listed for
# reinstateability but the active slate has both providers
# rejected (see Active slate below).

# 4. Pull the embedding model into Ollama
ollama pull bge-m3

# 5. Sanity tests (fast, no API calls)
make test

# 6. Run the pipeline
make step-0              # plumbing smoke; one toy-doc API call
make step-1              # model qualification smoke (~$22)
make step-2              # KV-cache verification per provider
make data-download       # QASPER + QuALITY + NovelQA → data/
make build-calibration   # 20 + 20 calibration pool, seed=42
make step-3-encoder      # encoder Recall@k against QASPER gold evidence
# Step 3 dry run + Step 4 variance + Step 5 hyperparameters: in progress
```

Outputs land under `outputs/` (gitignored) and `data/` (gitignored).
Verdict JSONs from each step go to `outputs/sanity/`. Cost-ledger
JSONL files go to `outputs/runs/<run_id>/ledger.jsonl`.

---

## Layout

```text
code/
├── configs/                       # YAML configs (every value carries source:)
│   ├── models.yaml                # active slate + rejected_candidates ledger
│   ├── methods.yaml               # lit-anchored hyperparameters with citations
│   ├── price_card.yaml            # provider rates, storage horizon, T=0 lock
│   └── embedding.yaml             # bge-m3 default + escalation chain
├── src/pilot/
│   ├── cli/
│   │   ├── step_0_smoke.py        # toy-doc plumbing smoke
│   │   ├── step_1_smoke.py        # model qualification smoke (resume-aware)
│   │   └── step_2_kvcache.py      # per-provider cache verification orchestrator
│   ├── data/
│   │   ├── download.py            # QASPER (S3) + QuALITY (zip) + NovelQA (HF)
│   │   └── build_calibration_pool.py  # deterministic 20+20 pool, seed=42
│   ├── prompts/                   # qa_freeform.txt, qa_multiplechoice.txt + loader
│   ├── providers/                 # AnswererProvider ABC + concrete adapters
│   │   ├── base.py
│   │   ├── anthropic_provider.py
│   │   ├── openai_provider.py
│   │   ├── gemini_provider.py
│   │   ├── dashscope_provider.py
│   │   ├── openai_compatible_provider.py  # OpenRouter + xAI subclasses
│   │   └── factory.py
│   ├── sanity/                    # KV-cache verification harness
│   ├── env.py                     # .env auto-loader (override=True)
│   ├── ledger.py                  # CostLedger (JSONL append-only)
│   ├── price_card.py              # USD computation from ledger rows
│   └── provenance.py              # source: field validator
├── tests/                         # 67 passing tests (no live API calls)
├── data/                          # local datasets (gitignored)
└── outputs/                       # ledgers, verdict JSONs (gitignored)
```

---

## Provenance rule (every locked value)

Every value in `configs/*.yaml` carries a `source:` field pointing at one of:

| Source kind | Example | What it means |
| --- | --- | --- |
| Pilot decision-matrix row | `pilot:5.8#12` | The N-runs decision per pilot plan § 5.8 row 12 |
| Literature citation key | `lit:sarthi2024raptor` | A citation in the thesis bibliography |
| Methodology rule | `rule:option-A` | A rule defined in `pilot_setup_plan.md` |
| Provider documentation | `provider:platform.claude.com/docs/...` | Pinned to a specific docs URL |
| Empirical pilot result | `empirical:step_2_kvcache_2026-05-02` | A measurement made during the pilot |

`tests/test_provenance.py` enforces that every leaf value in every
config file has a non-empty `source:` field. Violations fail the
test suite.

---

## Active slate snapshot (post-Step 2, 2026-05-02)

The active slate after Steps 1 and 2 is **10 candidates** across
4 providers. Sources of truth: `configs/models.yaml#closed_candidates`
and `configs/models.yaml#rejected_candidates` (with empirical
rejection reasons).

| Vendor | Candidates | Window | Cache verified |
| --- | --- | --- | --- |
| Google Gemini | 3.1 Pro Preview, 3.1 Flash Lite, flash-latest | 1M | yes (CachedContent) |
| xAI Grok | 4.3, 4-1-fast-non-reasoning, 4-fast-reasoning, 4.20-0309-non-reasoning, 4.20-0309-reasoning | 2M | yes (`x-grok-conv-id` required) |
| OpenRouter / DeepSeek | V4-Pro, V4-Flash | 1M | yes (`provider.only=["deepseek"]` pin required; only that upstream caches) |
| OpenRouter / Moonshot | Kimi K2.6 | 262k | **no** — zero of fourteen Moonshot endpoints support implicit caching; cost row falls back to fully-uncached pricing; QASPER-only |

Five rejected closed candidates are recorded with empirical
rejection reasons:

- Anthropic Sonnet 4.6 — rate-limit incompatible with multi-call workload (Step 2, 2026-05-02)
- Anthropic Opus 4.7 — empty response on 150k Lorem-Ipsum + same rate-limit cap (Step 1, 2026-05-02)
- OpenAI GPT-5.4 — account-tier blocks single 600k requests (Step 1, 2026-05-02)
- OpenAI GPT-5.5 — does not accept `temperature=0` (Step 1, 2026-05-02)
- OpenAI GPT-5.4-pro / GPT-5.5-pro — Responses-API only, not Chat Completions (Step 1, 2026-05-02)

All Anthropic and OpenAI candidates are reinstateable on a higher
account tier; the cache primitive itself is documented to work and
would not need re-verification.

---

## Pipeline reference

| Step | Make target | What it does | Live API spend (approx) |
| --- | --- | --- | --- |
| 0 | `make step-0` | One toy-doc call through one provider; smokes ledger + prompt + provider plumbing | <$0.01 |
| 1 | `make step-1` | 5k/150k/600k smoke across the slate; resume-aware (already-passed `(candidate, tier)` pairs are reused) | $22 across three runs |
| 2 | `make step-2` | Per-provider KV-cache verification on a 12k-token doc | <$1 |
| 3 | `make data-download` | Acquire QASPER + QuALITY + NovelQA into `data/` | $0 |
| 3 | `make build-calibration` | Deterministic 20+20 calibration pool, seed=42 | $0 |
| 3 | `make step-3-encoder` | Encoder Recall@k against QASPER gold evidence; locks BGE-M3 if Recall@20 ≥ 0.85 | $0 (local Ollama) |
| 3 | _pending_ | Summary-model paired build + 40-query end-to-end dry run | TBD |
| 4–7 | _pending_ | Variance, hyperparameters, cost-ledger sanity, lock-and-ship | TBD |

The pilot step CLIs are also installed as console scripts under the
venv: `pilot-step-0`, `pilot-kvcache`, etc. (See `pyproject.toml`
for the full list.)

### Resume mechanism (Step 1)

Step 1's orchestrator reads prior `step_1_smoke_*.json` verdict
files in `outputs/sanity/` and reuses any `(candidate, tier)` pair
whose status was `pass`. Re-runs only call the API for tiers that
failed or that involve newly-added candidates. Failed tiers are not
reused — they re-run on the next invocation, so bug fixes propagate
without forcing a full sweep.

---

## Caching: required adapter quirks

Step 2 uncovered three test-setup bugs that would silently destroy
the pilot's cache-amortization claim if left unfixed. Anyone
reproducing the pilot should be aware:

1. **OpenRouter requires per-model upstream pinning.** Sticky
   routing only activates _after_ a request is observed to use
   caching, so consecutive identical calls without a pin can land
   on different upstreams with disjoint cache stores. The
   `OpenRouterProvider` adapter pins `provider.only=["deepseek"]`
   for `deepseek/*` slugs. Slugs whose upstreams don't support
   caching (e.g. `moonshotai/kimi-k2.6`) are routed without a pin
   to avoid reducing availability.
2. **xAI requires `x-grok-conv-id` HTTP header.** The cache is
   per-server; without a stable conversation id the second call
   lands on a different server. The `XAIProvider` generates a
   per-instance UUID at construction and sends it on every call,
   so consecutive calls from the same adapter instance hit the
   same cache-warm server.
3. **Anthropic's `temperature` is deprecated on Opus 4.7.** Sending
   `temperature=0` to Opus 4.7 returns 400 BadRequest. The
   adapter omits both `temperature` and `top_p` for that model id
   and relies on model defaults.

---

## Datasets

All datasets land under `data/` (gitignored) and are not committed.

### QASPER

Acquired from Allen AI's S3 release tarballs (`qasper-train-dev-v0.3.tgz`,
`qasper-test-and-evaluator-v0.3.tgz`). The HuggingFace mirror
`allenai/qasper` is a script-based loader and breaks on `datasets`
v3+ — the script-loader path is unreliable.

Storage: `data/qasper/{train,dev,test,calibration_pool}.jsonl`.
Counts: 888 train / 281 dev / 416 test papers; 20-question
calibration pool drawn from dev with the
"requires at least one annotated evidence sentence" filter
applied so Evidence-F1 is computable.

### QuALITY

Acquired from the v1.0.1 release zip in the `nyu-mll/quality`
GitHub repo. Test labels are held by NYU; dev is the operational
evaluation set. The htmlstripped variant is used (HTML markup
replaced by line breaks; raw text only).

Storage: `data/quality/{train,dev,test}.jsonl`.
Counts: 300 train / 230 dev / 232 test (test labels stripped).

**Parsing gotcha**: QuALITY's htmlstripped JSONL contains Unicode
line separators (U+2028, U+2029) inside article text.
`str.splitlines()` over-splits on those and corrupts the JSONL
parse. The loader uses `str.split("\n")`.

### NovelQA

Acquired from the gated `NovelQA/NovelQA` HuggingFace dataset.
The `wangshelly/NovelQA` name in the original pilot plan was a
typo; the canonical mirror is `NovelQA/NovelQA`. The repo is a
flat file collection (no Parquet schema), so `load_dataset`
doesn't apply; the loader uses `huggingface_hub.snapshot_download`
plus zip extraction. Inside `NovelQA.zip`:
`Books/PublicDomain/B*.txt` (texts), `Data/PublicDomain/B*.json`
(questions per novel keyed by `Q####` ids).

Access: visit
[huggingface.co/datasets/NovelQA/NovelQA](https://huggingface.co/datasets/NovelQA/NovelQA),
click "Agree and access" on the dataset card with the same HF
account whose token is set in `HUGGINGFACE_ACCESS_TOKEN`.

The pilot evaluates only the public-domain subset (61 novels);
~28 copyright-protected novels are skipped because their texts
are withheld and end-to-end answer evaluation is not possible.
**B48** (_The History of Rome_, 2.58M tokens) is structurally
excluded from calibration sampling per pilot plan § 2.2 — kept
on disk for the outlier stress-test row only.

Storage: `data/novelqa/{full_texts/, questions.jsonl, calibration_pool.jsonl, calibration_novels.json, bookmeta.json}`.
Counts: 61 public-domain novels (≈15.4M tokens), 1,548 questions,
20-question calibration pool drawn from 4 calibration novels
(B01, B08, B41, B50) at seed=42.

---

## Sampling temperature

`T = 0` (greedy decoding) is fixed across all LLM calls in all
architectures, including preprocessing stages (RAPTOR
summarization, GraphRAG entity extraction + community summaries)
and the final answerer. Justification is in the thesis paper § 3.4.3
(Sampling Temperature) and rests on three pillars: literature
consensus on greedy decoding for QA evaluation, the falsifiability
requirement to hold sampling-temperature constant across
architectures, and the suppression of hallucinated labels in the
preprocessing index. Greedy decoding is _not_ deterministic
end-to-end (floating-point non-associativity, batch-invariance,
load balancing); the multi-run protocol in § 3.4.4 handles the
residual variance.

---

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

67 tests pass at HEAD. No test makes a live API call. The suite
covers:

- Cost-ledger roundtrip + USD computation
- Prompt template loader + deterministic rendering
- Provenance gate (every config value has `source:`)
- MC post-processor (`A`, `(A)`, `Option A`, full-text variants)
- Provider factory (each adapter exposes a uniform `ProviderResult`)
- Step 1 + Step 2 orchestrator behaviour with mocked providers

---

## Outputs and ledgers

- `outputs/sanity/step_*_*.json` — per-step verdict files (Step 1, Step 2).
- `outputs/runs/<run_id>/ledger.jsonl` — per-call cost ledger
  rows with `provider_request_id`, `prompt_hash`, `response_hash`,
  per-call wall-clock, token counts (uncached / cached / output),
  seed, temperature, top_p, max_tokens, provider region.
- `outputs/sanity/kvcache_<provider>_<timestamp>.json` — single-
  provider KV-cache verifier output (when invoked directly via
  `pilot-kvcache`).

All outputs are gitignored. Recoverable by re-running the
corresponding `make pilot-step-N` against the same configs and slate.

---

## License and data terms

- Code: MIT (see `pyproject.toml`).
- Configs and prompts: same as code (MIT).
- Datasets: each dataset retains its own license (QASPER per Allen AI
  release; QuALITY per nyu-mll repo terms; NovelQA Apache-2.0 with the
  HF dataset-card data-sharing agreement). The pilot scripts download
  data into `data/` (gitignored); no dataset content is redistributed
  through this repo.
- The `outputs/` and `data/` directories are excluded from version
  control to keep API-derived material and gated dataset content
  off the public repo.

---

## Related documents

All in the companion paper repo
[`BCJonkhout/thesis-msc-paper`](https://github.com/BCJonkhout/thesis-msc-paper):

- `notes/pilot_setup_plan.md` — full pilot procedure (Steps 0–7).
- `notes/pilot_findings.md` — empirical findings per pilot step.
- `notes/cache_ttl_per_provider.md` — provider cache TTL audit.
- `project.tex` — the thesis paper itself.
