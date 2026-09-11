<#
.SYNOPSIS
    Apollo audio restoration - rebuilds high frequencies lost to lossy codecs.

.DESCRIPTION
    Best choice for MP3/AAC/Opus-compressed material. Trained directly on
    codec artifacts, so it handles real compressed input rather than the
    clean lowpass AudioSR expects.

.EXAMPLE
    .\apollo.ps1 -Input "B:\music\track.mp3" -Output "B:\music\restored"

.EXAMPLE
    # process a whole folder
    .\apollo.ps1 -Input "B:\music\album" -Output "B:\music\album_restored"
#>
param(
    [Parameter(Mandatory = $true)][Alias('i')][string]$InputPath,
    [Parameter(Mandatory = $true)][Alias('o')][string]$Output,
    [ValidateSet('mp3', 'universal')][string]$Model = 'mp3',
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot 'msst'
$py = "$root\.venv\Scripts\python.exe"
$repo = "$root\MSST"

$ckptMap = @{
    'mp3'       = "$root\checkpoints\apollo_mp3_jusperlee.bin"
    'universal' = "$root\checkpoints\apollo_universal_sr.ckpt"
}
$cfgMap = @{
    'mp3'       = "$repo\configs\config_apollo.yaml"
    'universal' = "$root\checkpoints\config_apollo_universal.yaml"
}

$ckpt = $ckptMap[$Model]
$cfg = $cfgMap[$Model]
if (-not (Test-Path $ckpt)) { throw "Checkpoint missing for '$Model': $ckpt" }
if (-not (Test-Path $cfg)) { throw "Config missing for '$Model': $cfg" }

# MSST reads a folder, not a single file. Stage single files into a temp dir.
$staged = $null
if (Test-Path $InputPath -PathType Leaf) {
    $staged = Join-Path $env:TEMP ("apollo_in_" + [guid]::NewGuid().ToString('N').Substring(0, 8))
    New-Item -ItemType Directory -Force $staged | Out-Null
    # Apollo needs PCM wav; transcode anything else through ffmpeg first.
    if ($InputPath -notmatch '\.wav$') {
        $wav = Join-Path $staged ([IO.Path]::GetFileNameWithoutExtension($InputPath) + '.wav')
        Write-Host "transcoding to wav..." -ForegroundColor DarkGray
        ffmpeg -y -loglevel error -i $InputPath -ar 44100 -c:a pcm_s16le $wav
    }
    else {
        Copy-Item $InputPath $staged
    }
    $inFolder = $staged
}
else {
    $inFolder = $InputPath
}

$argsList = @(
    'inference.py'
    '--model_type'; 'apollo'
    '--config_path'; $cfg
    '--start_check_point'; $ckpt
    '--input_folder'; $inFolder
    '--store_dir'; $Output
)
if ($Cpu) { $argsList += '--force_cpu' }

Push-Location $repo
try {
    # MSST writes tqdm progress to stderr. Under ErrorActionPreference='Stop'
    # that gets promoted to a terminating NativeCommandError and aborts a run
    # that actually succeeded, so judge success by exit code instead.
    $ErrorActionPreference = 'Continue'
    & $py @argsList
    $code = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = 'Stop'
    Pop-Location
    if ($staged -and (Test-Path $staged)) { Remove-Item $staged -Recurse -Force }
}

if ($code -ne 0) { throw "Apollo inference failed (exit $code)" }
Write-Host "`nDone -> $Output" -ForegroundColor Green
