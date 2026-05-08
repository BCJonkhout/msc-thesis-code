#!/bin/bash
# Crash-recovery resume helper for the Phase F.1 v2 sweep.
#
# Run this after a laptop crash / hibernate / power loss to pick up
# the in-flight Phase F.1 v2 sweep (full 4-architecture × QASPER 20Q
# on Flash Lite, fixed RAPTOR + GraphRAG) where it left off.
#
# What it does:
#   1. Verifies Ollama is up (BGE-M3 embedder is required for
#      Naive RAG, RAPTOR, GraphRAG retrieval). Starts it in the
#      background if not.
#   2. Finds the most recent run dir matching the Phase F.1 v2
#      configuration: 4-arch, QASPER, gemini-3.1-flash-lite-preview,
#      literature prompt.
#   3. Re-launches step_3_dry_run with --resume-from pointing at that
#      run dir. Already-completed (arch, paper, qid) cells are
#      replayed from disk; only the missing cells re-execute.
#
# Idempotent — safe to run repeatedly. Each run produces a new
# self-contained run dir that includes everything from the chain.

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="outputs/sanity"
RUNS_DIR="outputs/runs"
PYTHON=".venv/Scripts/python.exe"
EXPECTED_ANSWERER="gemini-3.1-flash-lite-preview"

# ---- 1. Ollama liveness ----
if ! curl -s -o /dev/null --connect-timeout 3 -w "%{http_code}" \
        http://localhost:11434/api/tags | grep -q "200"; then
  echo "[resume] Ollama not up; starting it in the background"
  ollama serve > /tmp/ollama.log 2>&1 &
  until curl -s -o /dev/null --connect-timeout 1 -w "%{http_code}" \
            http://localhost:11434/api/tags | grep -q "200"; do
    sleep 1
  done
  echo "[resume] Ollama ready"
fi

# ---- 2. Find most recent matching run dir ----
# A "matching" run is one whose verdict JSON records the four
# architectures, QASPER dataset, the Flash-Lite answerer, and the
# literature prompt. We scan verdict JSONs in reverse-time order.
FOUND_RUN=""
for verdict in $(ls -t "$LOG_DIR"/step_3_dry_run_*.json 2>/dev/null); do
  if "$PYTHON" -c "
import json, sys
v = json.load(open(r'$verdict'))
ok = (
    set(v.get('architectures', [])) == {'flat', 'naive_rag', 'raptor', 'graphrag'}
    and v.get('datasets') == ['qasper']
    and v.get('answerer_model') == '$EXPECTED_ANSWERER'
    and v.get('prompt_style') == 'literature'
)
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    FOUND_RUN=$("$PYTHON" -c "
import json
v = json.load(open(r'$verdict'))
print(v.get('predictions_dir', ''))
" 2>/dev/null)
    break
  fi
done

if [ -z "$FOUND_RUN" ]; then
  echo "[resume] No prior matching run found; launching from scratch"
  RESUME_FLAG=""
else
  echo "[resume] Resuming from: $FOUND_RUN"
  RESUME_FLAG="--resume-from $FOUND_RUN"
fi

# ---- 3. Re-launch the sweep ----
exec env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$PYTHON" -m pilot.cli.step_3_dry_run \
    --architectures flat naive_rag raptor graphrag \
    --datasets qasper \
    --answerer-provider google \
    --answerer-model "$EXPECTED_ANSWERER" \
    --prompt-style literature \
    $RESUME_FLAG
