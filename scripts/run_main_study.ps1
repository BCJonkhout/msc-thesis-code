# Main-study launcher (Windows / native Ollama).
#
# Windows-native equivalent of scripts/run_main_study.sh. Use this when
# Ollama runs on the Windows host -- WSL2 has a separate network namespace
# and cannot reach the Windows localhost:11434, so `make main-study-slice`
# under WSL would try (and fail) to start its own Ollama. This script runs
# the Windows venv against the Windows Ollama directly, no WSL involved.
#
# Run from the code/ directory (or anywhere -- it cd's to the repo):
#   .\scripts\run_main_study.ps1 slice                 # dress rehearsal, primary only
#   .\scripts\run_main_study.ps1 full                  # full run (resumes over a slice)
#   .\scripts\run_main_study.ps1 slice -WithSecondary  # also run the grok slice (needs XAI_API_KEY)
#
# Idempotent / crash-safe: re-run to resume in place (same config -> same
# run dir; append-only ledger + per-chunk caches preserved). The slice
# shares the full run's dir, so `full` continues from the slice with no rework.
param(
    [ValidateSet('slice', 'full')]
    [string]$Mode = 'slice',
    [switch]$WithSecondary
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location (Join-Path $PSScriptRoot '..')

$python = '.\.venv\Scripts\python.exe'
$primary = 'gemini-3.1-flash-lite-preview'
$secondary = 'grok-4-fast-reasoning'
$summary = 'gemini-3.1-flash-lite-preview'

if (-not (Test-Path $python)) {
    Write-Error "venv python not found at $python. Run 'uv sync --extra test' first."
    exit 1
}

# Ollama liveness on the Windows host. We do NOT auto-start it -- manage
# Ollama yourself ('ollama serve' + 'ollama pull bge-m3'); this only checks
# it is reachable so the run doesn't fail deep into a build.
$code = 0
try {
    $code = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 `
            -Uri 'http://localhost:11434/api/tags').StatusCode
} catch { $code = 0 }
if ($code -ne 200) {
    Write-Error "Ollama not reachable at http://localhost:11434. Start it ('ollama serve') and ensure bge-m3 is pulled ('ollama pull bge-m3')."
    exit 1
}

$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_EMBED_CACHE_DIR = 'outputs/embed_cache'

$common = @(
    '--split', 'full', '--datasets', 'qasper', 'novelqa',
    '--architectures', 'flat', 'naive_rag', 'raptor', 'graphrag',
    '--summary-provider', 'google', '--summary-model', $summary,
    '--prompt-style', 'literature'
)
if ($Mode -eq 'slice') {
    $common += @('--max-docs-qasper', '50', '--max-docs-novelqa', '5')
    Write-Host '[main-study] MODE=slice (50 QASPER papers + 5 NovelQA novels)'
} else {
    Write-Host '[main-study] MODE=full (resumes in place over any prior slice)'
}

$base = @('-m', 'pilot.cli.step_3_dry_run') + $common

function Get-PredRowCount {
    $files = Get-ChildItem -Path 'outputs/runs/main-full-*/*_predictions.jsonl' -ErrorAction SilentlyContinue
    if (-not $files) { return 0 }
    return ($files | Get-Content -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
}

# Re-invoke the runner if it exits non-zero (a native crash like the UMAP
# segfault, an OOM, or a laptop crash). resume-in-place carries it forward
# over the already-completed cells. A no-progress guard stops a genuinely
# deterministic crash from looping forever instead of surfacing it.
function Invoke-RunWithResume {
    param([string[]]$RunArgs, [string]$Label, [int]$MaxAttempts = 12)
    $noProgress = 0
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        $before = Get-PredRowCount
        Write-Host "[main-study] $Label attempt $attempt/$MaxAttempts (predictions so far: $before)"
        & $python $RunArgs
        if ($LASTEXITCODE -eq 0) { return $true }
        $after = Get-PredRowCount
        Write-Warning "[main-study] $Label exited $LASTEXITCODE (native crash / interruption); predictions $before -> $after. Resuming in place."
        if ($after -le $before) {
            $noProgress++
            if ($noProgress -ge 2) {
                Write-Host "[main-study] $Label made NO progress across two retries -- stopping. Likely a deterministic crash on one document; inspect the last one logged above and build_failures.jsonl." -ForegroundColor Red
                return $false
            }
        } else {
            $noProgress = 0
        }
        Start-Sleep -Seconds 3
    }
    Write-Host "[main-study] $Label exhausted $MaxAttempts attempts." -ForegroundColor Red
    return $false
}

Write-Host "[main-study] === primary answerer: $primary (N=5) ==="
$primaryArgs = $base + @('--answerer-provider', 'google', '--answerer-model', $primary, '--num-runs', '5')
if (-not (Invoke-RunWithResume -RunArgs $primaryArgs -Label 'primary')) { exit 1 }

if ($WithSecondary) {
    Write-Host "[main-study] === secondary robustness slice: $secondary (N=1) ==="
    # --cache-required: grok reuses the primary's byte-identical retrieved
    # context (preprocess cache is keyed by summary model + encoder, not the
    # answerer). Requires XAI_API_KEY.
    $secondaryArgs = $base + @('--answerer-provider', 'xai', '--answerer-model', $secondary, '--run-index', '0', '--cache-required')
    if (-not (Invoke-RunWithResume -RunArgs $secondaryArgs -Label 'secondary')) { exit 1 }
} else {
    Write-Host '[main-study] secondary grok slice SKIPPED (-WithSecondary to include; needs XAI_API_KEY).'
}

Write-Host "[main-study] done ($Mode)."
Write-Host '[main-study] Track progress: wc -l outputs/runs/main-full-*/*_predictions.jsonl'
