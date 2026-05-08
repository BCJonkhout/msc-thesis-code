#!/bin/bash
# Phase F Pareto sweep — every active candidate × 4 architectures × QASPER 20Q.
#
# Runs sequentially so per-provider rate limits don't compound. Each
# candidate writes its own run dir; verdicts are aggregated post-hoc
# by `pilot.cli.phase_f_pareto`.
#
# Engineering preconditions:
#   - Multi-provider routing landed in commit b707855 (--summary-provider /
#     --summary-model flags). Required for non-Google candidates running
#     RAPTOR + GraphRAG; the summary stage stays on Gemini Flash Lite to
#     match the Step 3 asymmetric-tier finding.
#   - Step 4 set N = 5 for the main study, but Phase F sweeps stay at N = 1
#     because the cross-model rank-stability statistic is averaged across
#     architectures, not across runs of the same architecture.
#
# Cost envelope (extrapolated from Flash Lite measured at $0.126):
#   total ≈ $8.85; range $0.18 (DeepSeek Flash) to $3.78 (Grok 4.3).
#   Budget is recoverable — see `cost_report` after each candidate lands.

# Note: NO `set -e`. A single candidate's segfault (numba JIT cold-start
# crash, work item #3 in pilot Step 7 follow-ups) must not abort the whole
# sweep. Each candidate is wrapped in a retry-once-on-segfault loop;
# failures are logged and the sweep continues.

LOG_DIR="outputs/sanity"
mkdir -p "$LOG_DIR"

PYTHON=".venv/Scripts/python.exe"
DRY_RUN_FLAGS_4ARCH="--architectures flat naive_rag raptor graphrag --datasets qasper --prompt-style literature"
DRY_RUN_FLAGS_2ARCH="--architectures raptor graphrag --datasets qasper --prompt-style literature"

# Per pilot plan § 5.8 row #10: non-Google candidates pin the summary
# stage on Gemini Flash Lite via multi-provider routing so the asymmetric
# tier matches the canonical RAPTOR/GraphRAG production setup.
SUMMARY_FLAGS="--summary-provider google --summary-model gemini-3.1-flash-lite-preview"

# Candidates ordered cheapest-first. We skip Gemini 3.1 Pro Preview
# (already measured as the reference) and Flash Lite (Phase F.1, locked
# as primary). Grok 4.20-0309-non-reasoning is partial (Flat + Naive
# RAG already done in Phase F partial); only RAPTOR + GraphRAG remain.

_invoke_with_retry() {
  # Invoke the dry-run with one retry on numba JIT cold-start segfault
  # (exit 139). Subsequent invocations always succeed because numba's JIT
  # cache is warm by then. $1 = label, $2 = log file, rest = python args.
  local label=$1 logfile=$2; shift 2
  local attempt=1 max_attempts=2 ec
  while [ $attempt -le $max_attempts ]; do
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
      "$PYTHON" -m pilot.cli.step_3_dry_run "$@" >> "$logfile" 2>&1
    ec=$?
    if [ $ec -eq 0 ]; then
      return 0
    fi
    if [ $ec -eq 139 ] && [ $attempt -lt $max_attempts ]; then
      echo "  [$label] segfault on attempt $attempt; retrying..." | tee -a "$LOG_DIR/run_all_candidates.log"
      attempt=$((attempt + 1))
      continue
    fi
    return $ec
  done
}

run_full() {
  # $1 = label, $2 = provider, $3 = model, $4 = optional extra summary flags
  local label=$1 provider=$2 model=$3 extra=${4:-""}
  echo "===== $label / $provider / $model =====" | tee -a "$LOG_DIR/run_all_candidates.log"
  local logfile="$LOG_DIR/pareto_${label}.log"
  : > "$logfile"  # truncate per-candidate log on each attempt
  _invoke_with_retry "$label" "$logfile" \
    $DRY_RUN_FLAGS_4ARCH \
    --answerer-provider "$provider" --answerer-model "$model" \
    $extra
  echo "  [$label] exit=$?" | tee -a "$LOG_DIR/run_all_candidates.log"
}

run_partial_rg() {
  # RAPTOR + GraphRAG only (Grok 4.20-NR partial completion)
  local label=$1 provider=$2 model=$3 extra=${4:-""}
  echo "===== $label (RAPTOR + GraphRAG only) =====" | tee -a "$LOG_DIR/run_all_candidates.log"
  local logfile="$LOG_DIR/pareto_${label}.log"
  : > "$logfile"
  _invoke_with_retry "$label" "$logfile" \
    $DRY_RUN_FLAGS_2ARCH \
    --answerer-provider "$provider" --answerer-model "$model" \
    $extra
  echo "  [$label] exit=$?" | tee -a "$LOG_DIR/run_all_candidates.log"
}

# ---- 1. DeepSeek-V4-Flash (cheapest open-frontier; exercises multi-provider routing) ----
run_full "deepseek-v4-flash" "openrouter" "deepseek/deepseek-v4-flash" "$SUMMARY_FLAGS"

# ---- 2. Gemini Flash latest (Google mid-tier) ----
run_full "gemini-flash-latest" "google" "gemini-flash-latest"

# ---- 3. DeepSeek-V4-Pro (open-frontier flagship) ----
run_full "deepseek-v4-pro" "openrouter" "deepseek/deepseek-v4-pro" "$SUMMARY_FLAGS"

# ---- 4. Grok 4-1-fast-non-reasoning ----
run_full "grok-4-1-fast-nr" "xai" "grok-4-1-fast-non-reasoning" "$SUMMARY_FLAGS"

# ---- 5. Grok 4-fast-reasoning ----
run_full "grok-4-fast-reasoning" "xai" "grok-4-fast-reasoning" "$SUMMARY_FLAGS"

# ---- 6. Grok 4.20-0309-non-reasoning — RAPTOR + GraphRAG only (Phase F partial completion) ----
run_partial_rg "grok-4.20-0309-nr" "xai" "grok-4.20-0309-non-reasoning" "$SUMMARY_FLAGS"

# ---- 7. Grok 4.20-0309-reasoning ----
run_full "grok-4.20-0309-r" "xai" "grok-4.20-0309-reasoning" "$SUMMARY_FLAGS"

# ---- 8. Moonshot Kimi K2.6 (QASPER-only — 256k window cuts NovelQA) ----
run_full "kimi-k2.6" "openrouter" "moonshotai/kimi-k2.6" "$SUMMARY_FLAGS"

# ---- 9. Grok 4.3 (premium anchor; most expensive) ----
run_full "grok-4.3" "xai" "grok-4.3" "$SUMMARY_FLAGS"

echo "===== ALL CANDIDATES SWEEP COMPLETE =====" | tee -a "$LOG_DIR/run_all_candidates.log"
