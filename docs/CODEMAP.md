# CODEMAP — what produces what, and how it reaches the paper

This document is the provenance map for the code repository. It answers two
questions a reader of the published code cannot otherwise answer:

1. **Which script produced each number, table, and figure in the paper?**
2. **Which code is on the live main-study path, and which is pilot-era code kept
   only as a reproducibility record?**

The companion paper repository is
[`BCJonkhout/thesis-msc-paper`](https://github.com/BCJonkhout/thesis-msc-paper).
The paper benchmarks four QA architectures — **flat full-context, naive RAG,
RAPTOR, GraphRAG** — on two datasets (**QASPER**, local Answer-F1; **NovelQA**,
held-out Codabench multiple-choice gold) with a single answerer
(`gemini-3.1-flash-lite-preview`, T = 0) and a shared BGE-M3 encoder, N = 5
repeats. The headline result is that **flat full-context wins both datasets**
(identical ranking flat > naive_rag ≈ raptor > graphrag); RAPTOR and GraphRAG sit
off the cost–quality Pareto frontier.

> A note on history: the package is named `pilot/` for historical reasons. The
> project began as a small calibration pilot, which surfaced and fixed three
> confounds (architecture prompt mis-routing, abstention-template fallthrough,
> and a consensus-oracle NovelQA proxy). The pilot is retained in the paper as a
> methodology-validation step. The pilot-era analysis scripts are kept here as
> the reproducibility record behind that validation; the **canonical** scripts
> below are the ones that produce the paper's current claims.

---

## The result artifacts are sacred

Everything under `outputs/` and `data/` is a produced result or an input dataset
and is **never** deleted or hand-edited. Both directories are gitignored, so they
live only on the machine that produced them — but they are the ground truth that
every paper number traces back to. Any script that produced one of these
artifacts is part of the reproducibility chain and is kept even when a later
variant supersedes it.

Key on-disk artifacts (under `outputs/main_study/`):

| File | Produced by | Holds |
| --- | --- | --- |
| `scored_cells.jsonl` | `build_scored_cells.py` | one scored cell per (dataset, cluster, qid, run_index, arch) — the foundation every downstream number reads |
| `significance.json` | `significance_main_study.py` | per-pair clustered-bootstrap significance |
| `cost_per_arch.json`, `breakeven.json` | `breakeven_main_study.py` | per-arch deployment cost and break-even N* under both price cards |
| `memorization_control.json` | `novelqa_nocontext_control.py` + `qasper_nocontext_control.py` | closed-book (no-context) accuracy floors |

---

## The main-study pipeline (canonical — feeds the current paper)

```
            raw run dir (outputs/main_study/, gitignored)
                         │
   novelqa_codabench_accuracy.py ──┐   (recovers NovelQA accuracy from a Codabench submission)
                                   ▼
            build_scored_cells.py ──►  scored_cells.jsonl   ◄── foundation
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                            ▼
 significance_main_study.py   breakeven_main_study.py    {novelqa,qasper}_nocontext_control.py
   → significance.json        → cost_per_arch.json         → memorization_control.json
                              → breakeven.json
        └──────────────────────────┼───────────────────────────┘
                                   ▼
        tables_main_study.py            figures_main_study.py
          (reads the JSONs above)         (reads the JSONs above)
                                   ▼
              outputs/main_study/export/   (paper-named .tex + .pdf/.png)
                                   │
            make export-assets → scripts/export/promote_assets.sh
                                   ▼
                  thesis-msc/generated/mainstudy_*
                                   │
                       project.tex \input{generated/...}
```

### Producer → paper asset

`tables_main_study.py` writes the LaTeX tables; `figures_main_study.py` writes the
figures; both stage into `outputs/main_study/export/`, which
`make export-assets` promotes verbatim into `thesis-msc/generated/`. The paper
then `\input`s / `\includegraphics`es them (no metric is ever hardcoded in the
`.tex`).

| Paper asset (`thesis-msc/generated/`) | Produced by | Used in `project.tex` as |
| --- | --- | --- |
| `mainstudy_quality_ci.tex` | `tables_main_study.py` | `tab:results-arch-mean` (per-arch quality + 95% CI) |
| `mainstudy_significance.tex` | `tables_main_study.py` | `tab:results-significance` (paired clustered bootstrap) |
| `mainstudy_cost.tex` | `tables_main_study.py` | `tab:results-cost-quality` (deployment cost, both price cards) |
| `mainstudy_breakeven.tex` | `tables_main_study.py` | `tab:results-breakeven` (break-even N* per document) |
| `mainstudy_memorization.tex` | `tables_main_study.py` | closed-book control table |
| `mainstudy_accuracy_by_arch.{pdf,png}` | `figures_main_study.py` | per-arch accuracy figure (CIs = clustered bootstrap) |
| `mainstudy_pareto_cost_quality.{pdf,png}` | `figures_main_study.py` | cost–quality Pareto figure (the headline) |
| `mainstudy_breakeven_curves.{pdf,png}` | `figures_main_study.py` | amortised cost/query vs questions-per-document |

Supporting analysis input:

| Script | Role |
| --- | --- |
| `build_scored_cells.py` | scores every cell from the raw run dir; the single source the table/figure scripts read |
| `significance_main_study.py` | clustered paired bootstrap (paper clusters for QASPER, novel clusters for NovelQA) |
| `breakeven_main_study.py` | deployment cost + break-even, BGE-M3 GPU embed time billed at the measured local rate, under both price cards |
| `novelqa_codabench_accuracy.py` | recovers this study's NovelQA accuracy from a Codabench submission id |
| `novelqa_nocontext_control.py`, `qasper_nocontext_control.py` | closed-book memorization negative control on both datasets |
| `embedding_cost_calibration.py` | measures the local embed throughput (token volume × eval rate) that the cost model bills against |

---

## Pilot-era code (kept as a reproducibility record, not on the live path)

These scripts produced earlier (pilot / Phase G) results. Several were superseded
by the canonical scripts above; all are retained because they produced on-disk
artifacts and document how the confounds were found and fixed. They are **not**
the source of any current paper claim except where noted.

| Script(s) | What it did | Status |
| --- | --- | --- |
| `breakeven_analysis.py` | pilot per-(candidate, arch) break-even vs flat | superseded by `breakeven_main_study.py`; carries a known 5×-denominator bug — do not reuse for paper numbers |
| `kendall_arch_stability.py`, `…_rerun_20260519.py`, `…_rescored.py`, `…_under_gold_20260519.py` | Kendall τ_b architecture-rank stability across answerer models | pilot rank-stability series (consensus-oracle → re-scored → held-out gold lineage) |
| `kendall_cross_dataset_under_gold.py`, `kendall_stats_bootstrap.py` | cross-dataset τ_b and its bootstrap CI / permutation test | pilot; the τ_b material backs the paper's `tab:results-tau` (pilot-licensing appendix) |
| `novelqa_local_score.py`, `…_rerun_20260519.py`, `…_rescored.py` | NovelQA scoring against a cross-candidate consensus oracle | superseded by held-out Codabench gold (`novelqa_codabench_accuracy.py`) |
| `novelqa_reparse_predictions_rerun_20260519.py`, `…_rescored.py` | re-parse RAPTOR/GraphRAG MC predictions with the patched parser | pilot prediction-repair step |
| `collect_qasper_gold_20260520.py`, `rescore_gemini_flash_qasper_20260520.py` | QASPER rerun gold aggregation / rescore | pilot QASPER rerun |
| `rank_table.py`, `summarise_novelqa_grid.py` | pilot rank table and grid cost/completion summary | pilot reporting |
| `submit_rerun_to_codabench.py`, `submit_rescored_phaseg.py` | submit Phase G predictions to Codabench | pilot submission drivers |
| `figures/plot_dataset_interaction.py` | QASPER-vs-NovelQA scatter — depicts the **refuted** cross-dataset inversion | retired; output kept in `thesis-msc/figures/results/`, referenced by no current `.tex` |
| `figures/plot_per_arch_accuracy.py` | grouped per-arch accuracy bar chart | superseded by `figures_main_study.py`'s `accuracy_by_arch` |
| `figures/plot_kendall_distribution.py`, `figures/plot_rank_bump.py` | pilot τ_b histograms / rank-bump charts | retired pilot figures |

## Maintenance / dev utilities (produce no result)

| Script | Purpose |
| --- | --- |
| `check_tui.py` | self-test for the live progress display |
| `rekey_preprocess_cache.py` | re-key the on-disk preprocess cache to a new `code_version_hash` |

## Launchers (`scripts/*.sh`, `*.ps1`)

| Script | Purpose |
| --- | --- |
| `run_main_study.sh`, `run_main_study.ps1` | launch the four-architecture main study (POSIX / Windows) |
| `run_provider_lane.sh`, `run_provider_lane_novelqa.sh`, `chain_launch_novelqa.sh` | per-provider / per-dataset run lanes for the main study |
| `run_all_candidates.sh`, `resume_phase_f1_v2.sh` | pilot-era multi-candidate sweep / Phase-F resume |

---

## Library (`src/pilot/`)

The reusable plumbing the pipeline is built on. Highlights:

- `providers/` — `AnswererProvider` ABC + adapters; the active main-study answerer
  is the Gemini adapter. The Anthropic / OpenAI / DashScope adapters are retained
  for rejected candidates (see `configs/models.yaml#rejected_candidates`) so the
  slate is reinstateable, but are not on the main-study path.
- `architectures/` — `run_flat`, `run_naive_rag`, `run_raptor`, `run_graphrag`.
  `graphrag.py` is a **from-scratch** implementation; the Microsoft-`graphrag`
  shim that was considered and rejected is recorded in
  [`docs/graphrag_design_notes.md`](graphrag_design_notes.md).
- `ledger.py`, `price_card.py`, `provenance.py` — append-only cost ledger, USD
  computation, and the gate that every config value carries a `source:`.
- `eval/metrics.py`, `sanity/mc_postprocessor.py`, `codabench/` — scoring logic.
  Changing these changes holy result numbers, so they are edited for docs only.
- `encoders/` (BGE-M3 via Ollama, chunker), `preprocess_cache.py`,
  `build_call_cache.py` — the resumable build path.
- `third_party/raptor/` — vendored upstream with local patches; see its
  [`README.md`](../third_party/raptor/README.md).
