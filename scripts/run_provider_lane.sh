#!/bin/bash
# Per-provider sequential orchestrator for the v2.1 Pareto sweep.
#
# Three of these run in parallel — one per provider (Google, xAI,
# OpenRouter). Each is sequential within its vendor so rate limits
# aren't compounded, but across vendors the sweeps proceed
# concurrently.
#
# Each (label, provider, model) triplet runs the full 4-arch ×
# QASPER 20Q sweep with the post-audit-fix code. Non-Google
# answerers pin the summary stage on Gemini Flash Lite via the
# multi-provider routing flags landed in commit b707855.
#
# Usage:
#   bash scripts/run_provider_lane.sh <lane_name> <triplets...>
#
# Where each triplet is "label|provider|model".

LANE_NAME=$1
shift

LOG_DIR="outputs/sanity"
PYTHON=".venv/Scripts/python.exe"
DRY_RUN_FLAGS="--architectures flat naive_rag raptor graphrag --datasets qasper --prompt-style literature"

mkdir -p "$LOG_DIR"
DRIVER_LOG="$LOG_DIR/pareto_lane_${LANE_NAME}.log"

# Non-Google answerers use Gemini Flash Lite for summary (multi-provider
# routing landed in commit b707855). Google answerers don't need
# --summary-provider / --summary-model.
summary_flags_for() {
  case "$1" in
    google) echo "" ;;
    *) echo "--summary-provider google --summary-model gemini-3.1-flash-lite-preview" ;;
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

echo "===== LANE $LANE_NAME START ($(date -u +%Y-%m-%dT%H:%M:%SZ)) =====" | tee -a "$DRIVER_LOG"

for triplet in "$@"; do
  IFS='|' read -r label provider model <<< "$triplet"
  echo "----- $LANE_NAME : $label ($provider / $model) -----" | tee -a "$DRIVER_LOG"
  logfile="$LOG_DIR/pareto_v2_1_${label}.log"
  : > "$logfile"
  summary_flags=$(summary_flags_for "$provider")
  _invoke_with_retry "$label" "$logfile" \
    $DRY_RUN_FLAGS \
    --answerer-provider "$provider" --answerer-model "$model" \
    $summary_flags
  echo "  [$label] exit=$?" | tee -a "$DRIVER_LOG"
done

echo "===== LANE $LANE_NAME DONE ($(date -u +%Y-%m-%dT%H:%M:%SZ)) =====" | tee -a "$DRIVER_LOG"
