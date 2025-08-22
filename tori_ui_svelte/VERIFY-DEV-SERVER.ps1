Write-Host "=== VERIFYING DEV SERVER & ENDPOINTS ===" -ForegroundColor Cyan
Write-Host ""

# Wait a moment for server to be ready
Write-Host "Waiting 3 seconds for server startup..." -ForegroundColor Gray
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "[1/4] Checking dev server..." -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:5173" -Method Head -TimeoutSec 2 -ErrorAction Stop
    Write-Host "  [OK] Dev server responding" -ForegroundColor Green
} catch {
    Write-Host "  [X] Dev server not responding - make sure CLEAN-DEV-START.ps1 is running" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/4] Checking UA endpoint..." -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:5173/ua" -UseBasicParsing -TimeoutSec 2
    Write-Host "  [OK] UA endpoint working" -ForegroundColor Green
} catch {
    Write-Host "  [!] UA endpoint not responding" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[3/4] Checking device matrix endpoint..." -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:5173/device/matrix" -UseBasicParsing -TimeoutSec 2
    $data = $resp.Content | ConvertFrom-Json
    Write-Host "  [OK] Device matrix endpoint working" -ForegroundColor Green
    Write-Host "      Tier: $($data.tier)" -ForegroundColor Gray
    if ($data.caps) {
        Write-Host "      Max N: $($data.caps.maxN)" -ForegroundColor Gray
    }
} catch {
    Write-Host "  [!] Device matrix endpoint not responding" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[4/4] Checking hologram endpoint..." -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:5173/hologram" -Method Head -TimeoutSec 2 -ErrorAction Stop
    Write-Host "  [OK] Hologram endpoint exists" -ForegroundColor Green
} catch {
    Write-Host "  [!] Hologram endpoint not found (may need implementation)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " QUICK ACCESS URLS" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Visual Device Check:  http://localhost:5173/device/demo" -ForegroundColor White
Write-Host "Device Matrix API:    http://localhost:5173/device/matrix" -ForegroundColor White
Write-Host "User Agent Check:     http://localhost:5173/ua" -ForegroundColor White
Write-Host "Hologram View:        http://localhost:5173/hologram" -ForegroundColor White
Write-Host ""
Write-Host "[OK] Dev server verification complete!" -ForegroundColor Green