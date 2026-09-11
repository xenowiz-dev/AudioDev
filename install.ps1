<#
    AudioDev Studio installer.

    Rebuilds this box on a new machine: eight virtual environments, the model
    weights, and the checks that say whether it will actually run.

    Design notes, each learned the hard way here:

    * The package lists are FROZEN FROM A WORKING BOX (install\requirements\*.txt),
      not hand-written. Three different torch builds coexist -- cu128 for the
      generators, cu121 for the upscalers, cpu for the DSP venv -- and a box
      that installs the wrong one looks fine until a model loads.
    * Two Python versions. The upscalers need 3.10; everything else is 3.12.
      Installing them under one interpreter is how the original setup broke.
    * Model choice follows the CARD, not a constant. The XL variants need
      ~12 GB and are hidden on smaller GPUs unless -All is given, so a bigger
      machine gets more without anyone editing this file.
    * Everything is resumable and idempotent: an existing venv or a present
      model is skipped, so a failed run is re-run rather than unpicked.

    * The four upstream repos this depends on (ACE-Step, MSST, FlashSR, the
      AI-music detector) are NOT vendored into git. The `clones` stage puts
      them back at the exact commits this box was built against.

    Usage:
        git clone <repo> AudioDev; cd AudioDev
        .\install.ps1 -WhatIf              # say what would happen, change nothing
        .\install.ps1                      # prereqs + clones + venvs + core/standard models
        .\install.ps1 -Stage prereq        # just the checks
        .\install.ps1 -Stage models -All   # every model this card can run
        .\install.ps1 -SkipMiniMax         # 54 GB less
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateSet("all", "prereq", "clones", "venvs", "models", "verify")]
    [string]$Stage = "all",
    [string]$Root = "",
    [switch]$All,
    [switch]$SkipMiniMax,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
if (-not $Root) { $Root = $PSScriptRoot }   # wherever the repo was cloned
$Manifest = Join-Path $Root "install\models.json"
$ok = 0; $skip = 0; $fail = 0

function Say($msg, $colour = "Gray") { Write-Host $msg -ForegroundColor $colour }
function Head($msg) { Say ""; Say ("=" * 68) "DarkGray"; Say $msg "White"; Say ("=" * 68) "DarkGray" }
function Good($m) { $script:ok++;   Say "  OK    $m" "Green" }
function Skipd($m) { $script:skip++; Say "  skip  $m" "DarkGray" }
function Bad($m)  { $script:fail++; Say "  FAIL  $m" "Red" }

if (-not (Test-Path $Manifest)) { throw "manifest not found: $Manifest" }
$M = Get-Content $Manifest -Raw | ConvertFrom-Json

# ---------------------------------------------------------------- prereqs
function Get-VramGB {
    try {
        $o = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
        if ($o) { return [math]::Round(([int]($o -split "`n")[0]) / 1024, 1) }
    } catch { }
    return 0
}

function Find-Python($version) {
    # `py -3.12` is the reliable way on Windows; fall back to PATH.
    try {
        $p = & py "-$version" -c "import sys;print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
    } catch { }
    try {
        $p = (Get-Command "python$version" -ErrorAction SilentlyContinue).Source
        if ($p) { return $p }
    } catch { }
    return $null
}

function Stage-Prereq {
    Head "PREREQUISITES"
    $vram = Get-VramGB
    if ($vram -gt 0) { Good "NVIDIA GPU, $vram GB" } else { Bad "no NVIDIA GPU found (nvidia-smi)" }

    foreach ($v in @("3.12", "3.10")) {
        $p = Find-Python $v
        if ($p) { Good "Python $v -> $p" } else { Bad "Python $v missing (needed by some venvs)" }
    }
    foreach ($tool in @("ffmpeg", "ffprobe", "git")) {
        $c = Get-Command $tool -ErrorAction SilentlyContinue
        if ($c) { Good "$tool -> $($c.Source)" } else { Bad "$tool not on PATH" }
    }
    $free = [math]::Round((Get-PSDrive ($Root[0])).Free / 1GB, 1)
    $need = ($M.models | Where-Object { $_.tier -ne "optional" } | Measure-Object size_gb -Sum).Sum
    if ($free -gt $need) { Good "$free GB free on $($Root[0]): (need ~$need GB)" }
    else { Bad "$free GB free on $($Root[0]): but ~$need GB is needed" }
    return $vram
}

# ----------------------------------------------------------------- clones
function Stage-Clones {
    Head "UPSTREAM CLONES"
    foreach ($c in $M.clones) {
        $dest = Join-Path $Root $c.dest
        $short = $c.commit.Substring(0, 7)
        if (Test-Path (Join-Path $dest ".git")) {
            $have = (& git -C $dest rev-parse HEAD 2>$null)
            if ($have -eq $c.commit) { Skipd "$($c.dest) (at $short)" }
            else { Bad "$($c.dest) is at $($have.Substring(0,7)), expected $short -- git -C `"$dest`" checkout $($c.commit)" }
            continue
        }
        if ($PSCmdlet.ShouldProcess($c.dest, "git clone $($c.url) @ $short")) {
            try {
                & git clone --quiet $c.url $dest
                if ($LASTEXITCODE -ne 0) { Bad "$($c.dest): clone failed"; continue }
                & git -C $dest checkout --quiet $c.commit
                if ($LASTEXITCODE -eq 0) { Good "$($c.dest) @ $short  ($($c.what))" }
                else { Bad "$($c.dest): checkout of $short failed" }
            } catch { Bad "$($c.dest): $($_.Exception.Message)" }
        }
    }
}

# ------------------------------------------------------------------ venvs
function Venv-Dir($v) {
    # Most venvs live at <name>\.venv; a few (the labelling one) say where.
    if ($v.dir) { return Join-Path $Root $v.dir }
    return Join-Path $Root "$($v.name)\.venv"
}

function Stage-Venvs {
    Head "VIRTUAL ENVIRONMENTS"
    foreach ($v in $M.venvs) {
        $dir = Venv-Dir $v
        $py  = Join-Path $dir "Scripts\python.exe"
        $req = Join-Path $Root "install\requirements\$($v.name).txt"

        if ((Test-Path $py) -and -not $Force) { Skipd "$($v.name) (exists)"; continue }
        if (-not (Test-Path $req)) { Bad "$($v.name): no frozen requirements at $req"; continue }

        $base = Find-Python $v.python
        if (-not $base) { Bad "$($v.name): Python $($v.python) not available"; continue }

        if ($PSCmdlet.ShouldProcess($v.name, "create venv ($($v.python)) and install $(@(Get-Content $req).Count) packages")) {
            try {
                & $base -m venv $dir
                & $py -m pip install --quiet --upgrade pip
                # torch FIRST, from its own index: installing it as a transitive
                # dependency pulls the default CUDA build and silently ignores
                # the one this venv needs.
                if ($v.torch) {
                    $idx = $M.torch_index.($v.torch)
                    $tline = (Get-Content $req | Where-Object { $_ -match "^torch==" }) | Select-Object -First 1
                    if ($tline) {
                        Say "        torch: $tline from $($v.torch)"
                        & $py -m pip install --quiet $tline --index-url $idx
                    }
                }
                & $py -m pip install --quiet -r $req
                $tdesc = if ($v.torch) { $v.torch } else { "none" }
                Good "$($v.name) ($($v.python), torch $tdesc)"
            } catch { Bad "$($v.name): $($_.Exception.Message)" }
        }
    }
}

# ----------------------------------------------------------------- models
function Stage-Models($vram) {
    Head "MODELS"
    $hfPy = Join-Path $Root "acestep\.venv\Scripts\python.exe"
    if (-not (Test-Path $hfPy)) { $hfPy = Join-Path $Root "studio\.venv\Scripts\python.exe" }
    if (-not (Test-Path $hfPy)) { Bad "no venv with huggingface_hub yet - run -Stage venvs first"; return }

    foreach ($m in $M.models) {
        if ($m.tier -eq "optional" -and -not $All) { Skipd "$($m.id) (optional; -All to include)"; continue }
        if ($SkipMiniMax -and $m.id -eq "minimax") { Skipd "$($m.id) (-SkipMiniMax)"; continue }
        if ($m.min_vram_gb -gt 0 -and $vram -gt 0 -and $m.min_vram_gb -gt $vram) {
            Skipd "$($m.id) needs ~$($m.min_vram_gb) GB VRAM, this card has $vram"
            continue
        }

        $dest = Join-Path $Root $m.dest
        $rt = if ($m.repo_type) { $m.repo_type } else { "model" }
        if ($m.hf_cache) {
            $probe = Join-Path $dest ("hub\models--" + ($m.repo -replace "/", "--"))
        } else { $probe = $dest }
        if ((Test-Path $probe) -and -not $Force) { Skipd "$($m.id) (present)"; continue }

        if ($PSCmdlet.ShouldProcess($m.id, "download $($m.repo) (~$($m.size_gb) GB)")) {
            Say "        $($m.repo) -> $dest  (~$($m.size_gb) GB)"
            # Optional `ignore`: glob patterns not worth the bytes (demo audio).
            $ig = if ($m.ignore) { ",ignore_patterns=" + (ConvertTo-Json @($m.ignore) -Compress) } else { "" }
            $code = if ($m.hf_cache) {
                # Cache layout: many repos share one HF_HOME, deduplicated.
                "import os;os.environ['HF_HOME']=r'$dest';" +
                "from huggingface_hub import snapshot_download;" +
                "snapshot_download('$($m.repo)',max_workers=4$ig)"
            } elseif ($m.file) {
                # One file out of a repo, saved under OUR name: the Apollo
                # checkpoints are called something else upstream.
                "import os,shutil;from huggingface_hub import hf_hub_download;" +
                "p=hf_hub_download('$($m.repo)','$($m.file)',repo_type='$rt');" +
                "os.makedirs(os.path.dirname(r'$dest'),exist_ok=True);shutil.copyfile(p,r'$dest')"
            } else {
                "from huggingface_hub import snapshot_download;" +
                "snapshot_download('$($m.repo)',local_dir=r'$dest',repo_type='$rt',max_workers=4)"
            }
            try {
                & $hfPy -c $code
                if ($LASTEXITCODE -eq 0) { Good "$($m.id)" } else { Bad "$($m.id) (exit $LASTEXITCODE)" }
            } catch { Bad "$($m.id): $($_.Exception.Message)" }
        }
    }
}

# ----------------------------------------------------------------- verify
function Stage-Verify {
    Head "VERIFY"
    foreach ($c in $M.clones) {
        $dest = Join-Path $Root $c.dest
        if (Test-Path (Join-Path $dest ".git")) { Good "$($c.dest) present" } else { Bad "$($c.dest) missing -- run -Stage clones" }
    }
    foreach ($v in $M.venvs) {
        $py = Join-Path (Venv-Dir $v) "Scripts\python.exe"
        if (-not (Test-Path $py)) { Bad "$($v.name): venv missing"; continue }
        if ($v.torch) {
            $out = & $py -c "import torch;print(torch.__version__, torch.cuda.is_available())" 2>$null
            if ($LASTEXITCODE -eq 0) { Good "$($v.name): torch $out" }
            else { Bad "$($v.name): torch does not import" }
        } else {
            $out = & $py -c "import fastapi,uvicorn;print('web ok')" 2>$null
            if ($LASTEXITCODE -eq 0) { Good "$($v.name): $out" } else { Bad "$($v.name): web deps missing" }
        }
    }
    $ck = Join-Path $Root "acestep\checkpoints"
    if (Test-Path $ck) {
        $variants = Get-ChildItem $ck -Directory | Where-Object { $_.Name -like "acestep-v15-*" }
        if ($variants) { Good "DiT variants: $((($variants).Name) -join ', ')" }
        else { Bad "no acestep-v15-* checkpoint found" }
    } else { Bad "no checkpoints directory" }

    Say ""
    Say "Start the studio with:  .\web\serve.ps1" "White"
    Say "Then open http://127.0.0.1:7862/  (or publish it with -Tailscale)" "White"
}

# ------------------------------------------------------------------- main
$vram = 0
if ($Stage -in @("all", "prereq")) { $vram = Stage-Prereq }
if ($vram -eq 0) { $vram = Get-VramGB }
if ($Stage -in @("all", "clones")) { Stage-Clones }
if ($Stage -in @("all", "venvs"))  { Stage-Venvs }
if ($Stage -in @("all", "models")) { Stage-Models $vram }
if ($Stage -in @("all", "verify")) { Stage-Verify }

Head "SUMMARY"
Say "  $ok ok, $skip skipped, $fail failed" $(if ($fail) { "Yellow" } else { "Green" })
exit $(if ($fail) { 1 } else { 0 })
