# Fetch only the components the diffusers ModularPipeline actually references.
#
# modular_model_index.json names exactly seven subfolders. Everything else in the
# repo belongs to the SGLang-Omni serving path, which needs two CUDA GPUs and is
# not usable on this machine:
#
#   qwen_7B/             18.5 GB   SGLang copy of the LLM
#   flowmatching_vae.pth  9.8 GB   raw VAE for the SGLang path
#   dav.pth               492 MB
#   assets/ figures/               demo wav + images
#
# Skipping those takes the pull from ~57 GB to ~27.5 GB.
#
# Downloads into the HF cache (HF_HOME below), NOT --local-dir: the per-component
# specs in modular_model_index.json record the *hub repo id*, so load_components()
# resolves against the cache. Using --local-dir would make it re-download.

# NOT "Stop": Windows PowerShell 5.1 wraps a native exe's stderr in an ErrorRecord,
# so hf's routine "unauthenticated requests" warning would abort the script even
# though the download exits 0. Correctness is checked via $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$env:HF_HOME = Join-Path $PSScriptRoot "hf"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$repo = "MiniMaxAI/MiniMax-Music3"

$patterns = @(
    "modular_model_index.json", "config.json", "README.md", "LICENSE",
    "scripts/*",
    "condition_encoder/*", "language_model/*", "rvq_depth_decoder/*",
    "scheduler/*", "tokenizer/*", "transformer/*", "vocoder/*"
)

# Each pattern needs its own --include. A bare PowerShell array splat expands to
# `--include p1 p2 p3 ...`, where --include swallows only p1 and the rest land as
# positional FILENAMES -- which silently drops the first pattern from the pull.
$argv = @($repo)
foreach ($p in $patterns) { $argv += "--include"; $argv += $p }

& (Join-Path $PSScriptRoot ".venv\Scripts\hf.exe") download @argv
if ($LASTEXITCODE -ne 0) { throw "download failed ($LASTEXITCODE)" }
Write-Output "DOWNLOAD COMPLETE"
