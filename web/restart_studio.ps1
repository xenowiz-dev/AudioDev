<#
    Restart the studio server -- but only when nothing is running.

    This exists because I killed a job the user was mid-way through by running
    the "is the lane idle?" check and the restart in a single command: the
    answer arrived after the kill, so knowing better did not help. A gate is
    only a gate if it can refuse.

    It also kills the WORKERS, not just uvicorn. A previous restart matched
    only `uvicorn*7862` and left an acestep worker orphaned, holding 6.2 GB of
    VRAM and 21.8 GB of RAM with nothing to talk to.

        .\restart_studio.ps1          # refuses if a job is running
        .\restart_studio.ps1 -Force   # only when you mean it
#>
param([switch]$Force)

$Api = "http://127.0.0.1:7862"
$Root = Split-Path -Parent $PSScriptRoot
$env:AUDIODEV_ROOT = $Root            # inherited by uvicorn and its workers
$Py  = Join-Path $Root "studio\.venv\Scripts\python.exe"
$Web = $PSScriptRoot

# --- the gate -----------------------------------------------------------
$active = $null
try {
    $r = Invoke-WebRequest -Uri "$Api/api/jobs?active=1" -UseBasicParsing -TimeoutSec 6
    $active = ($r.Content | ConvertFrom-Json).jobs
} catch {
    Write-Host "server is not answering -- treating as safe to (re)start."
}

if ($active -and $active.Count -gt 0 -and -not $Force) {
    Write-Host "REFUSED: $($active.Count) job(s) running:" -ForegroundColor Yellow
    foreach ($j in $active) {
        $live = if ($j.live) { $j.live.msg } else { $j.state }
        Write-Host "   $($j.id)  $($j.kind)  $live"
    }
    Write-Host ""
    Write-Host "Nothing was stopped. Wait for it, cancel it in the GPU sheet,"
    Write-Host "or re-run with -Force if you truly mean to kill it."
    exit 1
}
if ($active -and $active.Count -gt 0) {
    Write-Host "-Force given; killing $($active.Count) running job(s)." -ForegroundColor Red
}

# --- stop the server AND its workers ------------------------------------
$pat = '\*uvicorn\*7862\*|acestep_worker|minimax_worker'
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        $_.CommandLine -like "*uvicorn*7862*" -or
        $_.CommandLine -like "*acestep_worker*" -or
        $_.CommandLine -like "*minimax_worker*"
    } | ForEach-Object {
        Write-Host "  stopping $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Seconds 3

# --- start ---------------------------------------------------------------
Start-Process -FilePath $Py `
    -ArgumentList @("-m","uvicorn","api:app","--app-dir",$Web,
                    "--host","127.0.0.1","--port","7862","--log-level","info") `
    -RedirectStandardOutput "$Web\srv_log.txt" `
    -RedirectStandardError  "$Web\srv_err.txt" `
    -WindowStyle Hidden | Out-Null

for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    try {
        $h = (Invoke-WebRequest -Uri "$Api/api/health" -UseBasicParsing -TimeoutSec 3).Content
        Write-Host "up: $h"
        exit 0
    } catch { }
}
Write-Host "server did not come up; see $Web\srv_err.txt" -ForegroundColor Red
exit 1
