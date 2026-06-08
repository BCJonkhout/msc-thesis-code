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
# By default ONLY the primary answerer (Gemini Flash Lite) runs, which
# needs just GEMINI_API_KEY + local Ollama. The grok-4-fast-reasoning
# cross-vendor robustness slice (N=1) is a supplementary check and is
# OPT-IN -- it is the only thing that needs XAI_API_KEY:
#   WITH_SECONDARY=1 scripts/run_main_study.sh slice
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
# Honors OLLAMA_HOST so this can point at an Ollama on another host (e.g.
# the Windows host when run from WSL -- that server must bind 0.0.0.0 and
# the firewall must allow it). Fails fast instead of hanging forever when
# Ollama is neither reachable nor installable here.
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
case "$OLLAMA_URL" in http://*|https://*) ;; *) OLLAMA_URL="http://$OLLAMA_URL" ;; esac
if ! curl -s -o /dev/null --connect-timeout 3 -w "%{http_code}" \
        "$OLLAMA_URL/api/tags" | grep -q "200"; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "[main-study] ERROR: Ollama not reachable at $OLLAMA_URL and 'ollama' is" >&2
    echo "[main-study]   not on PATH here, so it cannot be started. If Ollama runs" >&2
    echo "[main-study]   on another host (e.g. the Windows host while this runs in" >&2
    echo "[main-study]   WSL), set OLLAMA_HOST to it (server must bind 0.0.0.0) -- or" >&2
    echo "[main-study]   run on the host where Ollama lives (scripts/run_main_study.ps1" >&2
    echo "[main-study]   on Windows)." >&2
    exit 1
  fi
  echo "[main-study] starting Ollama (OLLAMA_NUM_PARALLEL=1)"
  OLLAMA_NUM_PARALLEL=1 ollama serve > /tmp/ollama.log 2>&1 &
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null --connect-timeout 1 -w "%{http_code}" \
          http://localhost:11434/api/tags | grep -q "200"; then break; fi
    sleep 1
  done
  if ! curl -s -o /dev/null --connect-timeout 1 -w "%{http_code}" \
        http://localhost:11434/api/tags | grep -q "200"; then
    echo "[main-study] ERROR: Ollama did not come up within 60s; see /tmp/ollama.log" >&2
    exit 1
  fi
fi

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export OLLAMA_EMBED_CACHE_DIR=outputs/embed_cache

COMMON="--split full $CAPS --datasets qasper novelqa \
  --architectures flat naive_rag raptor graphrag \
  --summary-provider google --summary-model $SUMMARY \
  --prompt-style literature"

# Total prediction rows on disk -- used to detect whether a crashed
# attempt made forward progress before resuming.
pred_rows() {
  local n=0 f
  for f in outputs/runs/main-full-*/*_predictions.jsonl; do
    [ -f "$f" ] && n=$((n + $(wc -l < "$f")))
  done
  echo "$n"
}

# Re-invoke the runner if it exits non-zero (native crash / OOM / laptop
# crash); resume-in-place carries it forward over the completed cells. A
# no-progress guard stops a deterministic crash from looping instead of
# surfacing it.
run_with_resume() {  # $1=label, rest=runner args
  local label="$1"; shift
  local max=12 attempt noprog=0 before after
  for attempt in $(seq 1 "$max"); do
    before=$(pred_rows)
    echo "[main-study] $label attempt $attempt/$max (predictions so far: $before)"
    if "$PYTHON" -m pilot.cli.step_3_dry_run "$@"; then return 0; fi
    after=$(pred_rows)
    echo "[main-study] $label crashed/interrupted; resuming in place ($before -> $after)" >&2
    if [ "$after" -le "$before" ]; then
      noprog=$((noprog + 1))
      if [ "$noprog" -ge 2 ]; then
        echo "[main-study] $label made NO progress across two retries -- stopping (likely a deterministic crash on one document; check build_failures.jsonl)." >&2
        return 1
      fi
    else
      noprog=0
    fi
    sleep 3
  done
  echo "[main-study] $label exhausted $max attempts." >&2
  return 1
}

echo "[main-study] === primary answerer: $PRIMARY (N=5) ==="
# The primary pass populates the per-document preprocess cache and the
# per-chunk embed cache on miss (no --cache-required, so it builds).
# shellcheck disable=SC2086
run_with_resume "primary" $COMMON \
  --answerer-provider google --answerer-model "$PRIMARY" \
  --num-runs 5 || exit 1

if [ "${WITH_SECONDARY:-0}" = "1" ]; then
  echo "[main-study] === secondary robustness slice: $SECONDARY (N=1) ==="
  # The secondary reuses the primary's cached retrieved context: the
  # preprocess cache key is keyed by summary model + encoder, NOT the
  # answerer, so --cache-required guarantees grok sees byte-identical
  # context and the only difference is the answerer call. (--cache-required
  # also aborts loudly if the primary's builds are incomplete, so run this
  # only after the primary pass has finished.) Requires XAI_API_KEY.
  # shellcheck disable=SC2086
  run_with_resume "secondary" $COMMON \
    --answerer-provider xai --answerer-model "$SECONDARY" \
    --run-index 0 --cache-required || exit 1
else
  echo "[main-study] secondary grok robustness slice SKIPPED."
  echo "[main-study]   (opt in with WITH_SECONDARY=1; that step is the only one needing XAI_API_KEY.)"
fi

echo "[main-study] done ($MODE)."
echo "[main-study] Record the exact grok-4-fast-reasoning build id in the run manifest."
echo "[main-study] Before scoring a FULL run, gate on completeness:"
echo "  make completeness-check RUN=<primary_run_dir_name> NUM_RUNS=5 DATASETS=\"qasper novelqa\""
