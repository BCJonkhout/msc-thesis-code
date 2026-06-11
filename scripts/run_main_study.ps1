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

# Config preflight: required API keys must be set BEFORE we spend hours
# building. GEMINI_API_KEY is always required (primary answerer + summaries);
# XAI_API_KEY only when the grok cross-vendor slice (-WithSecondary) is
# requested. If a key is missing, offer to paste it in-session (hidden) so the
# user doesn't have to abort and re-run.
function Read-SecretToEnv {
    param([string]$Prompt)
    $sec = Read-Host -AsSecureString $Prompt
    $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    try { return [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr) }
    finally { [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}
function Test-RunConfig {
    param([bool]$WithSecondary)
    if ([string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
        Write-Host '[main-study] GEMINI_API_KEY is not set (required: Gemini answerer + summaries).' -ForegroundColor Yellow
        $v = Read-SecretToEnv -Prompt 'Paste your GEMINI_API_KEY (or just press Enter to abort)'
        if ([string]::IsNullOrWhiteSpace($v)) { Write-Error 'GEMINI_API_KEY not provided; aborting.'; return $false }
        $env:GEMINI_API_KEY = $v.Trim()
    }
    if ($WithSecondary -and [string]::IsNullOrWhiteSpace($env:XAI_API_KEY)) {
        Write-Host '[main-study] -WithSecondary set but XAI_API_KEY is not set (the grok slice needs it).' -ForegroundColor Yellow
        $v = Read-SecretToEnv -Prompt 'Paste your XAI_API_KEY (or just press Enter to abort)'
        if ([string]::IsNullOrWhiteSpace($v)) { Write-Error 'XAI_API_KEY not provided; aborting.'; return $false }
        $env:XAI_API_KEY = $v.Trim()
    }
    $msg = '[main-study] Config OK: GEMINI_API_KEY set'
    if ($WithSecondary) { $msg += '; XAI_API_KEY set' }
    Write-Host "$msg." -ForegroundColor Green
    return $true
}
if (-not (Test-RunConfig -WithSecondary:$WithSecondary)) { exit 1 }

# Ollama liveness on the Windows host. If it is not up, try to START it for
# the user ('ollama serve' in the background) and pull the bge-m3 embedder if
# it is missing -- a multi-day run should not die on the doorstep because the
# embedder server wasn't running yet. If it cannot be auto-started (Ollama not
# on PATH, or the start fails) it falls back to prompting and retrying.
function Test-Ollama {
    param([string]$Url, [string]$Model)
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri "$Url/api/tags"
        if ($r.StatusCode -eq 200) {
            return @{ Up = $true; HasModel = ($r.Content -match [regex]::Escape($Model)) }
        }
    } catch {}
    return @{ Up = $false; HasModel = $false }
}
function Initialize-Ollama {
    param([string]$Url = 'http://localhost:11434', [string]$Model = 'bge-m3')
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    while ($true) {
        $s = Test-Ollama -Url $Url -Model $Model
        if ($s.Up -and $s.HasModel) {
            Write-Host "[main-study] Ollama is up and '$Model' is available." -ForegroundColor Green
            return $true
        }
        # Start the server ourselves if it is installed and not up.
        if (-not $s.Up -and $ollama) {
            Write-Host "[main-study] Ollama not running; starting 'ollama serve' in the background..." -ForegroundColor Cyan
            $env:OLLAMA_NUM_PARALLEL = '1'
            $env:OLLAMA_MAX_LOADED_MODELS = '1'
            Start-Process -FilePath $ollama.Source -ArgumentList 'serve' -WindowStyle Hidden | Out-Null
            for ($i = 0; $i -lt 30; $i++) {
                Start-Sleep -Seconds 1
                if ((Test-Ollama -Url $Url -Model $Model).Up) { break }
            }
            $s = Test-Ollama -Url $Url -Model $Model
        }
        # Pull the embedder if the server is up but the model is missing.
        if ($s.Up -and -not $s.HasModel -and $ollama) {
            Write-Host "[main-study] Pulling embedder '$Model' (one-time)..." -ForegroundColor Cyan
            & $ollama.Source pull $Model
            $s = Test-Ollama -Url $Url -Model $Model
        }
        if ($s.Up -and $s.HasModel) {
            Write-Host "[main-study] Ollama is up and '$Model' is available." -ForegroundColor Green
            return $true
        }
        # Could not auto-start/pull -> prompt the user and retry.
        Write-Host ''
        if (-not $s.Up) {
            Write-Host "[main-study] Could not reach or auto-start Ollama at $Url." -ForegroundColor Yellow
            Write-Host '  Start it in another terminal:  ollama serve'
        }
        else {
            Write-Host "[main-study] Ollama is up but '$Model' is not available." -ForegroundColor Yellow
            Write-Host "  Pull it in another terminal:   ollama pull $Model"
        }
        $ans = Read-Host 'Press Enter to retry once it is ready, or type q then Enter to abort'
        if ($ans -eq 'q') { return $false }
    }
}
if (-not (Initialize-Ollama)) {
    Write-Error 'Ollama not ready; aborting the run.'
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
# N protocol: N=5 per cell (pre-registered). Builds run once (cached); the
# five answer passes reuse them, so N=5 is answer-only on top of the
# one-time build cost. Question-level bootstrap CIs + paired architecture
# tests are computed over the N=5 grid at scoring time.
$numRuns = 5
if ($Mode -eq 'slice') {
    $common += @('--max-docs-qasper', '50', '--max-docs-novelqa', '5')
    Write-Host '[main-study] MODE=slice (rehearsal: 50 QASPER papers + 5 NovelQA novels, N=5)'
} else {
    Write-Host '[main-study] MODE=full (N=5 over all questions)'
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
$primaryArgs = $base + @('--answerer-provider', 'google', '--answerer-model', $primary, '--num-runs', "$numRuns")
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
