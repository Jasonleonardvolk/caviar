# TORI Simple Dev Install for Windows
param(
    [switch]$Headless = $false
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TORI ONE-COMMAND DEV INSTALL" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Green
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.(\d+)") {
    $minorVersion = [int]$matches[1]
    if ($minorVersion -lt 8) {
        Write-Host "ERROR: Python 3.8+ required (found: $pythonVersion)" -ForegroundColor Red
        exit 1
    }
    Write-Host "OK: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}

# Create venv
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Green
if (Test-Path .venv) {
    Write-Host "   Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv .venv
    Write-Host "OK: Virtual environment created" -ForegroundColor Green
}

# Activate venv
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1
Write-Host "OK: Virtual environment activated" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Green
if (Test-Path pyproject.toml) {
    python -m pip install -e ".[dev]" --quiet
    Write-Host "OK: Dependencies installed from pyproject.toml" -ForegroundColor Green
} else {
    python -m pip install -r requirements-dev.txt --quiet
    Write-Host "OK: Dependencies installed from requirements-dev.txt" -ForegroundColor Green
}

# Final message
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Launch TORI
if (-not $Headless) {
    Write-Host "Launching TORI..." -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
    Write-Host ""
    python enhanced_launcher.py
} else {
    Write-Host "Headless mode - skipping launch" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
