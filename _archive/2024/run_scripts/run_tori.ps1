#!/usr/bin/env pwsh
# Run TORI with enhanced launcher

Write-Host "`n=== Starting TORI System ===" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# Ensure we're in the right directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if poetry environment is active
try {
    $poetryCheck = poetry env info 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[INFO] Activating Poetry environment..." -ForegroundColor Yellow
        poetry shell
    }
} catch {
    Write-Host "[WARNING] Poetry not found. Ensure you have activated the virtual environment." -ForegroundColor Yellow
}

# Run the enhanced launcher
Write-Host "`n[*] Starting enhanced launcher..." -ForegroundColor Yellow
poetry run python enhanced_launcher.py

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[OK] TORI exited cleanly" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] TORI exited with error code: $LASTEXITCODE" -ForegroundColor Red
}
