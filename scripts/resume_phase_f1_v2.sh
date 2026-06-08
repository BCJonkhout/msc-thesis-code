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
#   2. Re-launches step_3_dry_run with the Phase F.1 v2 configuration
#      (4-arch, QASPER, gemini-3.1-flash-lite-preview, literature
#      prompt). The run dir is derived deterministically from that
#      config (canonical_run_id), so the runner resumes IN PLACE:
#      already-completed (arch, paper, qid, run_index) cells are
#      skipped and only the missing cells re-execute.
#
# Idempotent — safe to run repeatedly. The same config always maps to
# the same run dir; the append-only ledger keeps every prior cost row.

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="outputs/sanity"
RUNS_DIR="outputs/runs"
PYTHON=".venv/Scripts/python.exe"
EXPECTED_ANSWERER="gemini-3.1-flash-lite-preview"

# ---- 1. Ollama liveness ----
if ! curl -s -o /dev/null --connect-timeout 3 -w "%{http_code}" \
        http://localhost:11434/api/tags | grep -q "200"; then
  echo "[resume] Ollama not up; starting it in the background (single-flight)"
  # OLLAMA_NUM_PARALLEL=1 caps server-side embed concurrency so parallel
  # build lanes queue rather than saturate the single-threaded embedder
  # (the B41 sustained-500 root cause). If Ollama is already running this
  # branch is skipped, so the cap only applies to a helper-started server.
  OLLAMA_NUM_PARALLEL=1 ollama serve > /tmp/ollama.log 2>&1 &
  until curl -s -o /dev/null --connect-timeout 1 -w "%{http_code}" \
            http://localhost:11434/api/tags | grep -q "200"; do
    sleep 1
  done
  echo "[resume] Ollama ready"
fi

# ---- 2. Re-launch the sweep (resume-in-place) ----
# The run directory is derived deterministically from the configuration,
# so re-invoking with the SAME config reopens the same dir and resumes
# in place. No verdict scan and no --resume-from needed — idempotent by
# construction.
echo "[resume] Resuming in place by config (canonical run dir)"
# OLLAMA_EMBED_CACHE_DIR persists per-chunk embeddings so a crash
# mid-build re-embeds only the unfinished tail, not the whole document.
exec env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  OLLAMA_EMBED_CACHE_DIR=outputs/embed_cache \
  "$PYTHON" -m pilot.cli.step_3_dry_run \
    --architectures flat naive_rag raptor graphrag \
    --datasets qasper \
    --answerer-provider google \
    --answerer-model "$EXPECTED_ANSWERER" \
    --prompt-style literature
