<#
    AudioDev Studio web API -- 127.0.0.1:7862

    Loopback only, on purpose: `tailscale serve --bg 7862` puts it on the
    tailnet with TLS, and binding 127.0.0.1 is what keeps Windows from raising
    a firewall prompt. The old gradio app keeps :7861 to itself.

        .\serve.ps1                 # run it
        .\serve.ps1 -Port 7893      # a second instance for tests
        .\serve.ps1 -Reload         # dev: uvicorn's StatReload, watching web\
        .\serve.ps1 -Tailscale      # also publish it over tailscale serve
#>
param(
    [int]$Port = 7862,
    [string]$BindHost = "127.0.0.1",
    [switch]$Reload,
    [switch]$Tailscale,
    [switch]$TestHooks
)

$ErrorActionPreference = "Stop"

$Root   = Split-Path -Parent $PSScriptRoot
$env:AUDIODEV_ROOT = $Root            # inherited by uvicorn and its workers
$Python = Join-Path $Root "studio\.venv\Scripts\python.exe"
$Web    = $PSScriptRoot

if (-not (Test-Path $Python)) { throw "studio venv python not found: $Python" }
if (-not (Test-Path $Web))    { throw "web root not found: $Web" }

# The studio venv has gradio, and therefore fastapi + uvicorn already.
$args = @("-m", "uvicorn", "api:app", "--app-dir", $Web,
          "--host", $BindHost, "--port", "$Port", "--log-level", "info")
if ($Reload) {
    # --reload watches the CWD, not --app-dir, so name the directory.
    $args += @("--reload", "--reload-dir", $Web)
}

if ($TestHooks) { $env:STUDIO_TEST_HOOKS = "1" }

if ($Tailscale) {
    Write-Host "publishing over tailscale serve on $Port ..."
    & tailscale serve --bg $Port
}

Write-Host "AudioDev Studio API  ->  http://$BindHost`:$Port/"
& $Python @args
