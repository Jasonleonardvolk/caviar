# TORI Dev Install - Clean Version
param([switch]$Headless = $false)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "TORI ONE-COMMAND DEV INSTALL" -ForegroundColor Cyan
Write-Host ""

# Check Python
$pythonCmd = $null
foreach ($cmd in @('python', 'python3', 'py')) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match '^Python\s+3\.(\d+)') {
            if ([int]$matches[1] -ge 8) {
                $pythonCmd = $cmd
                Write-Host "Found Python: $ver" -ForegroundColor Green
                break
            }
        }
    } catch { }
}

if (-not $pythonCmd) {
    Write-Host "ERROR: Python 3.8+ not found" -ForegroundColor Red
    exit 1
}

# Create venv
if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    & $pythonCmd -m venv .venv
}

# Activate venv
Write-Host "Activating venv..." -ForegroundColor Yellow
. .\.venv\Scripts\Activate.ps1

# Install deps
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install -U pip wheel setuptools --quiet
if (Test-Path pyproject.toml) {
    python -m pip install -e ".[dev]"
} else {
    python -m pip install -r requirements-dev.txt
}

Write-Host ""
Write-Host "INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host ""

if (-not $Headless) {
    Write-Host "Launching TORI..." -ForegroundColor Green
    python enhanced_launcher.py
}
