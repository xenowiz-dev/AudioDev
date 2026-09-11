# Launch AudioDev Studio.
#   PC    : http://127.0.0.1:7861
#   phone : https://bonelab.tailb09e67.ts.net   (any device on the tailnet)
#
# The phone URL needs no per-launch setup: `tailscale serve --bg 7861` was
# configured once and persists across reboots, proxying tailnet HTTPS to
# localhost. The app itself stays bound to 127.0.0.1.
#
# NOT "Stop": Windows PowerShell 5.1 wraps a native exe's stderr in an
# ErrorRecord, and gradio logs routine startup lines there.
$ErrorActionPreference = "Continue"

$env:AUDIODEV_ROOT = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") (Join-Path $PSScriptRoot "app.py") @args
