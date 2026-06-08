#!/bin/bash
# Main-study launcher.
#
# Runs the single-answerer main study over the full evaluation split:
#   - primary answerer gemini-3.1-flash-lite-preview at N=5, then
#   - the grok-4-fast-reasoning cross-vendor robustness slice at N=1.
# Split = full: QASPER dev minus calibration (papers with >=2 questions)
# and all public-domain NovelQA novels except B48.
#
# Usage:
#   scripts/run_main_study.sh slice   # dress rehearsal: 50 papers + 5 novels
#   scripts/run_main_study.sh full    # full run (resumes IN PLACE over a prior slice)
#
# Idempotent and crash-safe. Re-run after a laptop crash to resume in
# place: the same config maps to the same run dir, the append-only ledger
# keeps every cost row, the per-chunk embed cache and per-document
# preprocess cache mean a re-run re-embeds/re-builds only the unfinished
# tail, and completed (arch, paper, question, run_index) cells are
# skipped. The validation slice shares the full run's dir, so after the
# dress rehearsal `... full` continues from exactly where the slice
# stopped without redoing anything.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-slice}"
PYTHON=".venv/Scripts/python.exe"
[ -x "$PYTHON" ] || PYTHON=".venv/bin/python"

PRIMARY="gemini-3.1-flash-lite-preview"
SECONDARY="grok-4-fast-reasoning"
SUMMARY="gemini-3.1-flash-lite-preview"

CAPS=""
if [ "$MODE" = "slice" ]; then
  CAPS="--max-docs-qasper 50 --max-docs-novelqa 5"
  echo "[main-study] MODE=slice (dress rehearsal: 50 QASPER papers + 5 NovelQA novels)"
elif [ "$MODE" = "full" ]; then
  echo "[main-study] MODE=full (resumes in place over any prior slice)"
else
  echo "usage: $0 {slice|full}" >&2
  exit 2
fi

# Ollama liveness (BGE-M3 embedder). Single-flight server-side concurrency
# avoids the sustained-500 saturation that blocked the B41 build.
if ! curl -s -o /dev/null --connect-timeout 3 -w "%{http_code}" \
        http://localhost:11434/api/tags | grep -q "200"; then
  echo "[main-study] starting Ollama (OLLAMA_NUM_PARALLEL=1)"
  OLLAMA_NUM_PARALLEL=1 ollama serve > /tmp/ollama.log 2>&1 &
  until curl -s -o /dev/null --connect-timeout 1 -w "%{http_code}" \
            http://localhost:11434/api/tags | grep -q "200"; do
    sleep 1
  done
fi

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export OLLAMA_EMBED_CACHE_DIR=outputs/embed_cache

COMMON="--split full $CAPS --datasets qasper novelqa \
  --architectures flat naive_rag raptor graphrag \
  --summary-provider google --summary-model $SUMMARY \
  --prompt-style literature"

echo "[main-study] === primary answerer: $PRIMARY (N=5) ==="
# The primary pass populates the per-document preprocess cache and the
# per-chunk embed cache on miss (no --cache-required, so it builds).
# shellcheck disable=SC2086
$PYTHON -m pilot.cli.step_3_dry_run $COMMON \
  --answerer-provider google --answerer-model "$PRIMARY" \
  --num-runs 5

echo "[main-study] === secondary robustness slice: $SECONDARY (N=1) ==="
# The secondary reuses the primary's cached retrieved context: the
# preprocess cache key is keyed by summary model + encoder, NOT the
# answerer, so --cache-required guarantees grok sees byte-identical
# context and the only difference is the answerer call. (--cache-required
# also aborts loudly if the primary's builds are incomplete, so run this
# only after the primary pass has finished.)
# shellcheck disable=SC2086
$PYTHON -m pilot.cli.step_3_dry_run $COMMON \
  --answerer-provider xai --answerer-model "$SECONDARY" \
  --run-index 0 --cache-required

echo "[main-study] done ($MODE)."
echo "[main-study] Record the exact grok-4-fast-reasoning build id in the run manifest."
echo "[main-study] Before scoring a FULL run, gate on completeness:"
echo "  make completeness-check RUN=<primary_run_dir_name> NUM_RUNS=5 DATASETS=\"qasper novelqa\""
