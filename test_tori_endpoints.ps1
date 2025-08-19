#!/usr/bin/env pwsh
# Test TORI endpoints in PowerShell

Write-Host "`n=== Testing TORI Endpoints ===" -ForegroundColor Cyan

# Test API Health
Write-Host "`n[1] Testing API Health..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/health'
    Write-Host "[OK] API Health: $($health | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] API Health failed: $_" -ForegroundColor Red
}

# Test MCP Status
Write-Host "`n[2] Testing MCP Status..." -ForegroundColor Yellow
try {
    $status = Invoke-RestMethod -Method Get -Uri 'http://localhost:8100/api/system/status'
    Write-Host "[OK] MCP Status: $($status | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] MCP Status failed: $_" -ForegroundColor Red
}

# Rebuild Lattice
Write-Host "`n[3] Rebuilding Lattice..." -ForegroundColor Yellow
try {
    $rebuild = Invoke-RestMethod -Method Post -Uri 'http://localhost:8002/api/lattice/rebuild?full=true'
    Write-Host "[OK] Lattice Rebuild: $($rebuild | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Lattice Rebuild failed: $_" -ForegroundColor Red
    Write-Host "Note: This endpoint might not exist. Check your logs for oscillator counts instead." -ForegroundColor Yellow
}

Write-Host "`n=== Done ===" -ForegroundColor Cyan
