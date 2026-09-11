<#
.SYNOPSIS
    AudioSR - diffusion-based audio super-resolution, any input -> 48 kHz.

.DESCRIPTION
    Uses the chunked wrapper (audiosr_long.py) so full-length tracks work.
    Stock AudioSR batches the entire file at once and OOMs past ~20 s on a
    10 GB card.

    On a ground-truth test (real track -> 96 kbps -> restore -> compare against
    the original) this was the MOST ACCURATE of the three installed tools:
    4.37 dB mean error vs Apollo's 7.00 and FlashSR's 11.34. It is also the
    slowest by ~10x. Use it when quality matters, Apollo when throughput does.

    IMPORTANT: AudioSR was trained on lowpass-filtered audio only. Feed it a
    raw MP3 and it hallucinates from codec artifacts. Always pass -AutoLowpass
    for lossy sources - it detects the real cutoff and filters there first,
    and that is how the winning number above was produced.

.EXAMPLE
    .\audiosr.ps1 -i "track.wav" -o "track_48k.wav"

.EXAMPLE
    # lossy source - detect and lowpass at the real cutoff first
    .\audiosr.ps1 -i "track.mp3" -o "track_sr.wav" -AutoLowpass

.EXAMPLE
    # faster / lower VRAM
    .\audiosr.ps1 -i "in.wav" -o "out.wav" -DdimSteps 25 -Chunk 5.12
#>
param(
    [Parameter(Mandatory = $true)][Alias('i')][string]$InputPath,
    [Parameter(Mandatory = $true)][Alias('o')][string]$Output,
    [ValidateSet('basic', 'speech')][string]$Model = 'basic',
    # 5.12 s is the verified default on a 10 GB card. AudioSR pads each window
    # up to the next 2.5 s multiple, so 10.24 really means a 12.5 s window.
    # Raise it for fewer seams only when the GPU is otherwise idle.
    [double]$Chunk = 5.12,
    [double]$Overlap = 1.0,
    [int]$DdimSteps = 50,
    [double]$GuidanceScale = 3.5,
    [int]$Seed = 42,
    [switch]$AutoLowpass
)

$ErrorActionPreference = 'Stop'
$py = Join-Path $PSScriptRoot 'audiosr\.venv\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'audiosr\audiosr_long.py'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

# audiosr_long reads via soundfile, which has no mp3 decoder on all builds.
# Transcode non-wav input through ffmpeg first.
$tmp = $null
$src = $InputPath
if ($InputPath -notmatch '\.(wav|flac)$') {
    $tmp = Join-Path $env:TEMP ("audiosr_in_" + [guid]::NewGuid().ToString('N').Substring(0, 8) + ".wav")
    Write-Host "transcoding to wav..." -ForegroundColor DarkGray
    ffmpeg -y -loglevel error -i $InputPath -ar 44100 -c:a pcm_s16le $tmp
    $src = $tmp
}

$argsList = @(
    $script
    '-i'; $src
    '-o'; $Output
    '--chunk'; $Chunk
    '--overlap'; $Overlap
    '--ddim_steps'; $DdimSteps
    '-gs'; $GuidanceScale
    '--seed'; $Seed
    '--model_name'; $Model
)
if ($AutoLowpass) { $argsList += '--auto-lowpass' }

try {
    # AudioSR writes DDIM progress bars and deprecation warnings to stderr.
    # Under ErrorActionPreference='Stop' those abort a successful run, so
    # judge success by exit code instead.
    $ErrorActionPreference = 'Continue'
    & $py @argsList
    $code = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = 'Stop'
    if ($tmp -and (Test-Path $tmp)) { Remove-Item $tmp -Force }
}

if ($code -ne 0) { throw "AudioSR failed (exit $code)" }
Write-Host "`nDone -> $Output" -ForegroundColor Green
