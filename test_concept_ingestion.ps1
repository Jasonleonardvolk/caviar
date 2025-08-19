#!/usr/bin/env pwsh
# Test concept ingestion and lattice rebuild

Write-Host "`n=== Testing Concept Ingestion ===" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# Test 1: Check concept mesh stats
Write-Host "`n[1] Checking concept mesh stats..." -ForegroundColor Yellow
try {
    $stats = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/concept-mesh/stats'
    Write-Host "[OK] Concept mesh stats:" -ForegroundColor Green
    Write-Host "  Total concepts: $($stats.totalConcepts)" -ForegroundColor White
    Write-Host "  Average strength: $($stats.averageStrength)" -ForegroundColor White
    Write-Host "  Last update: $($stats.lastUpdate)" -ForegroundColor White
    
    if ($stats.totalConcepts -eq 0) {
        Write-Host "[WARNING] Concept mesh is empty!" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to get concept mesh stats: $_" -ForegroundColor Red
}

# Test 2: Rebuild lattice
Write-Host "`n[2] Rebuilding lattice (full rebuild)..." -ForegroundColor Yellow
try {
    $rebuild = Invoke-RestMethod -Method Post -Uri 'http://localhost:8002/api/lattice/rebuild?full=true'
    Write-Host "[OK] Lattice rebuild result:" -ForegroundColor Green
    Write-Host "  Success: $($rebuild.success)" -ForegroundColor White
    Write-Host "  Mode: $($rebuild.mode)" -ForegroundColor White
    Write-Host "  Oscillators created: $($rebuild.oscillators_created)" -ForegroundColor White
    Write-Host "  Total oscillators: $($rebuild.total_oscillators)" -ForegroundColor White
    
    if ($rebuild.oscillators_created -eq 0) {
        Write-Host "[WARNING] No oscillators created! Check concept mesh loading." -ForegroundColor Red
    } else {
        Write-Host "[SUCCESS] Created $($rebuild.oscillators_created) oscillators!" -ForegroundColor Green
    }
} catch {
    Write-Host "[ERROR] Failed to rebuild lattice: $_" -ForegroundColor Red
}

# Test 3: Check lattice snapshot
Write-Host "`n[3] Checking lattice snapshot..." -ForegroundColor Yellow
try {
    $snapshot = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/lattice/snapshot'
    Write-Host "[OK] Lattice snapshot:" -ForegroundColor Green
    Write-Host "  Oscillators: $($snapshot.summary.oscillators)" -ForegroundColor White
    Write-Host "  Order parameter: $($snapshot.summary.order_parameter)" -ForegroundColor White
    Write-Host "  Phase entropy: $($snapshot.summary.phase_entropy)" -ForegroundColor White
} catch {
    Write-Host "[ERROR] Failed to get lattice snapshot: $_" -ForegroundColor Red
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan
