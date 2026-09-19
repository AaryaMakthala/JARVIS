# JARVIS Phase 0 CI-style check script: ruff + mypy(policy) + pytest + verify_env.
# Usage:  .\scripts\check.ps1          (from the repository root or anywhere)
#         .\scripts\check.ps1 -SkipFormat
param(
    [switch]$SkipFormat
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSCommandPath | Split-Path -Parent
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) {
    Write-Host "[FAIL] .venv missing - create it with: python -m venv .venv" -ForegroundColor Red
    exit 1
}

Write-Host "==> Python version check"
& $py -c "import sys; v=sys.version_info[:2]; assert (3,11) <= v < (3,13), f'need Python 3.11/3.12, have {sys.version}'; import struct; assert sys.maxsize > 2**32"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipFormat) {
    Write-Host "==> ruff format --check"
    & $py -m ruff format --check .
    if ($LASTEXITCODE -ne 0) { Write-Host "run: .venv\Scripts\python.exe -m ruff format ." -ForegroundColor Yellow; exit $LASTEXITCODE }
}

Write-Host "==> ruff check"
& $py -m ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> mypy (policy is strict)"
& $py -m mypy src/jarvis/policy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> pytest (no windows_only / voice / slow)"
& $py -m pytest -q -m "not windows_only and not voice and not slow"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> verify_env --phase 1"
& $py scripts\verify_env.py --phase 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[ OK ] all JARVIS checks passed" -ForegroundColor Green