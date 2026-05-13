#!/bin/bash
# Chain-launch the NovelQA Phase G full-grid sweep after the QASPER
# Pareto lanes finish.
#
# The QASPER lanes (currently running, tasks b3vi08js2 / bkw275r8s /
# b00s6bj22) write `===== LANE <name> DONE` to their driver logs
# when they finish. This script polls those log files; once all three
# QASPER lanes are marked DONE, it launches the 3 NovelQA lanes in
# parallel.
#
# Rationale: Ollama is the shared bottleneck (BGE-M3 embedder for
# Naive RAG / RAPTOR / GraphRAG retrieval). Running 6 parallel
# step_3_dry_run processes against single-threaded Ollama would
# trigger queue waits and 500s. The chain ensures Ollama load
# stays at 3-way max.
#
# Idempotent: re-run after a laptop crash to pick up wherever the
# chain stopped. step_3_dry_run's --resume-from is NOT used here
# because the v2.1 QASPER and NovelQA runs are separate sweeps; the
# chain just orchestrates timing.

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="outputs/sanity"
QASPER_LANES=(google xai openrouter)
NOVELQA_LANES=(google xai openrouter)

echo "[chain] Waiting for QASPER lanes to finish: ${QASPER_LANES[*]}"

all_done() {
  for lane in "${QASPER_LANES[@]}"; do
    if ! grep -q "===== LANE $lane DONE" "$LOG_DIR/pareto_lane_${lane}.log" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

while ! all_done; do
  sleep 30
done

echo "[chain] All QASPER lanes finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[chain] Launching NovelQA Phase G lanes in parallel..."

# Google lane: Pro + Flash Lite + Flash latest (3 candidates).
bash scripts/run_provider_lane_novelqa.sh google \
  "g_novelqa-gemini-3.1-flash-lite-preview|google|gemini-3.1-flash-lite-preview" \
  "g_novelqa-gemini-3.1-pro-preview|google|gemini-3.1-pro-preview" \
  "g_novelqa-gemini-flash-latest|google|gemini-flash-latest" \
  > "$LOG_DIR/lane_novelqa_google.driver.log" 2>&1 &
GOOGLE_PID=$!

# xAI lane: 5 Grok variants.
bash scripts/run_provider_lane_novelqa.sh xai \
  "g_novelqa-grok-4-1-fast-non-reasoning|xai|grok-4-1-fast-non-reasoning" \
  "g_novelqa-grok-4-fast-reasoning|xai|grok-4-fast-reasoning" \
  "g_novelqa-grok-4.20-0309-non-reasoning|xai|grok-4.20-0309-non-reasoning" \
  "g_novelqa-grok-4.20-0309-reasoning|xai|grok-4.20-0309-reasoning" \
  "g_novelqa-grok-4.3|xai|grok-4.3" \
  > "$LOG_DIR/lane_novelqa_xai.driver.log" 2>&1 &
XAI_PID=$!

# OpenRouter lane: DeepSeek flash + pro. Kimi excluded (NovelQA-rejected
# per configs/models.yaml#moonshotai/kimi-k2.6: 256k < 780k threshold).
bash scripts/run_provider_lane_novelqa.sh openrouter \
  "g_novelqa-deepseek-v4-flash|openrouter|deepseek/deepseek-v4-flash" \
  "g_novelqa-deepseek-v4-pro|openrouter|deepseek/deepseek-v4-pro" \
  > "$LOG_DIR/lane_novelqa_openrouter.driver.log" 2>&1 &
OPENROUTER_PID=$!

echo "[chain] NovelQA lanes launched: google PID=$GOOGLE_PID, xai PID=$XAI_PID, openrouter PID=$OPENROUTER_PID"

wait $GOOGLE_PID $XAI_PID $OPENROUTER_PID
echo "[chain] All NovelQA lanes done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
