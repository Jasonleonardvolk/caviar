# Quick-Start.ps1 - Simplest possible mock server start

Write-Host "Quick Start Mock Server" -ForegroundColor Cyan
Write-Host ""

# Kill ALL node processes (nuclear option)
Write-Host "Stopping any existing node processes..." -ForegroundColor Yellow
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Go to directory
Set-Location D:\Dev\kha\tori_ui_svelte

# Build if needed
if (-not (Test-Path "build\index.js")) {
    Write-Host "Building..." -ForegroundColor Yellow
    pnpm run build
}

# Start with mock environment
Write-Host "Starting on http://localhost:3000" -ForegroundColor Green
Write-Host ""

$env:IRIS_USE_MOCKS = "1"
$env:IRIS_ALLOW_UNAUTH = "1"
$env:PORT = "3000"

node build\index.js
