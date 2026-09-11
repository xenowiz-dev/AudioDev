<#
    Scheduled LoRA training. Registered with schtasks; see register_task.ps1.

    Deliberately thin: everything that decides anything lives in run_lora.py,
    so a scheduled run and a hand-typed one take exactly the same path. This
    only adds what a scheduled run needs and an interactive one does not --
    a log to read in the morning, and a status file that says what happened
    without having to parse it.

    The GPU gate still applies. A run that starts against a busy card does not
    fail cleanly, it thrashes for hours, so "5pm" means "5pm if the card is
    actually free" and a refusal is a good outcome, not a failure.
#>
param(
    [string]$Preset = "auto",           # scales with the card; see run_lora.py
    [switch]$SkipPreprocess
)

$ErrorActionPreference = "Continue"
$Root   = Split-Path -Parent $PSScriptRoot
$env:AUDIODEV_ROOT = $Root
$Train  = $PSScriptRoot
$Py     = Join-Path $Root "studio\.venv\Scripts\python.exe"
$Stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$Log    = Join-Path $Train "logs\train-$Stamp.log"
$Status = Join-Path $Train "logs\last_run.json"

New-Item -ItemType Directory -Force -Path (Join-Path $Train "logs") | Out-Null

function Write-Status($state, $detail) {
    @{
        state   = $state
        detail  = $detail
        preset  = $Preset
        log     = $Log
        started = $Stamp
        ended   = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    } | ConvertTo-Json | Set-Content -Path $Status -Encoding utf8
}

"=== scheduled LoRA run $Stamp ===" | Tee-Object -FilePath $Log
"preset: $Preset" | Tee-Object -FilePath $Log -Append

# --- the gate, before anything expensive -------------------------------------
& $Py (Join-Path $Train "run_lora.py") check 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) {
    "GATE BLOCKED - nothing started." | Tee-Object -FilePath $Log -Append
    Write-Status "blocked" "the GPU was not free at 17:00; see the log for which check failed"
    exit 1
}

# --- preprocess only if the tensors are missing ------------------------------
$tensors = Join-Path $Train "preprocessed"
if (-not $SkipPreprocess -and
    (-not (Test-Path $tensors) -or -not (Get-ChildItem $tensors -ErrorAction SilentlyContinue))) {
    "preprocessing (no tensors found)" | Tee-Object -FilePath $Log -Append
    & $Py (Join-Path $Train "run_lora.py") preprocess 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Status "failed" "preprocess failed"
        exit 1
    }
} else {
    "tensors present - skipping preprocess" | Tee-Object -FilePath $Log -Append
}

# --- train -------------------------------------------------------------------
$t0 = Get-Date
& $Py (Join-Path $Train "run_lora.py") train --preset $Preset 2>&1 |
    Tee-Object -FilePath $Log -Append
$code = $LASTEXITCODE
$mins = [int]((Get-Date) - $t0).TotalMinutes

if ($code -eq 0) {
    "DONE in $mins min" | Tee-Object -FilePath $Log -Append
    Write-Status "done" "$mins minutes; adapters are in train\lora_out and appear in the studio's LoRA dropdown automatically"
} else {
    "FAILED (exit $code) after $mins min" | Tee-Object -FilePath $Log -Append
    Write-Status "failed" "exit $code after $mins minutes"
}
exit $code
