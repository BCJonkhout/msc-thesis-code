#!/bin/bash
# Per-provider sequential orchestrator for Phase G — NovelQA full
# 4-architecture grid.
#
# Three of these run in parallel — one per provider (Google, xAI,
# OpenRouter). Each is sequential within its vendor so rate limits
# aren't compounded; across vendors the sweeps proceed concurrently.
#
# Differs from the QASPER orchestrator (run_provider_lane.sh) in
# three ways:
#
#   1. --datasets novelqa (NOT qasper)
#   2. NovelQA labels are leaderboard-only — predictions are written
#      to the run dir for Codabench submission, but the verdict's
#      per_arch_macro is empty (no local F1 to report).
#   3. Kimi K2.6 is excluded from NovelQA per
#      configs/models.yaml#moonshotai/kimi-k2.6: 256k window cannot
#      fit full NovelQA novels in Flat full-context mode.
#
# Usage:
#   bash scripts/run_provider_lane_novelqa.sh <lane_name> <triplets...>
# Each triplet: "label|provider|model".

LANE_NAME=$1
shift

LOG_DIR="outputs/sanity"
PYTHON=".venv/Scripts/python.exe"

# Full 4-architecture sweep — Option A "fine enough" full-grid per
# the methodology discussion: avoids the Phase-G.1 → G.2 cherry-
# picking risk by running every (architecture × candidate) cell.
DRY_RUN_FLAGS="--architectures flat naive_rag raptor graphrag --datasets novelqa --prompt-style literature"

# Non-Google answerers pin the summary stage on Gemini Flash Lite via
# multi-provider routing (commit b707855). Required for RAPTOR + GraphRAG
# preprocessing on candidates whose own provider doesn't have a cheap
# summary tier.
SUMMARY_FLAGS_NON_GOOGLE="--summary-provider google --summary-model gemini-3.1-flash-lite-preview"

mkdir -p "$LOG_DIR"
DRIVER_LOG="$LOG_DIR/pareto_lane_novelqa_${LANE_NAME}.log"

summary_flags_for() {
  case "$1" in
    google) echo "" ;;
    *) echo "$SUMMARY_FLAGS_NON_GOOGLE" ;;
  esac
}

_invoke_with_retry() {
  local label=$1 logfile=$2; shift 2
  local attempt=1 max_attempts=2 ec
  while [ $attempt -le $max_attempts ]; do
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
      "$PYTHON" -m pilot.cli.step_3_dry_run "$@" >> "$logfile" 2>&1
    ec=$?
    [ $ec -eq 0 ] && return 0
    if [ $ec -eq 139 ] && [ $attempt -lt $max_attempts ]; then
      echo "  [$label] segfault on attempt $attempt; retrying..." | tee -a "$DRIVER_LOG"
      attempt=$((attempt + 1))
      continue
    fi
    return $ec
  done
}

echo "===== NOVELQA LANE $LANE_NAME START ($(date -u +%Y-%m-%dT%H:%M:%SZ)) =====" | tee -a "$DRIVER_LOG"

for triplet in "$@"; do
  IFS='|' read -r label provider model <<< "$triplet"

  # Kimi K2.6 is NovelQA-rejected (configs/models.yaml).
  # Skip with a logged note rather than failing the lane.
  if [ "$model" = "moonshotai/kimi-k2.6" ]; then
    echo "----- SKIP $label : Kimi K2.6 NovelQA-rejected (256k < 780k threshold) -----" | tee -a "$DRIVER_LOG"
    continue
  fi

  echo "----- $LANE_NAME : $label ($provider / $model) -----" | tee -a "$DRIVER_LOG"
  logfile="$LOG_DIR/pareto_novelqa_${label}.log"
  : > "$logfile"
  summary_flags=$(summary_flags_for "$provider")
  _invoke_with_retry "$label" "$logfile" \
    $DRY_RUN_FLAGS \
    --answerer-provider "$provider" --answerer-model "$model" \
    $summary_flags
  echo "  [$label] exit=$?" | tee -a "$DRIVER_LOG"
done

echo "===== NOVELQA LANE $LANE_NAME DONE ($(date -u +%Y-%m-%dT%H:%M:%SZ)) =====" | tee -a "$DRIVER_LOG"
