#!/usr/bin/env pwsh
# Test Soliton Init Endpoint
# ==========================

Write-Host "`n=== Testing Soliton Init Endpoint ===" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# Test 1: Check if API is running
Write-Host "`n[1] Checking API health..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/health'
    Write-Host "[OK] API is healthy" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] API is not responding on port 8002" -ForegroundColor Red
    Write-Host "Please ensure the API server is running: poetry run python enhanced_launcher.py" -ForegroundColor Yellow
    exit 1
}

# Test 2: Test Soliton health endpoint
Write-Host "`n[2] Checking Soliton health..." -ForegroundColor Yellow
try {
    $solitonHealth = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/soliton/health'
    Write-Host "[OK] Soliton health check passed:" -ForegroundColor Green
    Write-Host "  Status: $($solitonHealth.status)" -ForegroundColor White
    Write-Host "  Engine: $($solitonHealth.engine)" -ForegroundColor White
} catch {
    Write-Host "[WARNING] Soliton health check failed: $_" -ForegroundColor Yellow
}

# Test 3: Test Soliton init with correct payload
Write-Host "`n[3] Testing Soliton init endpoint..." -ForegroundColor Yellow
$initPayload = @{
    user_id = "adminuser"
    lattice_reset = $false
} | ConvertTo-Json

Write-Host "Sending payload:" -ForegroundColor Gray
Write-Host $initPayload -ForegroundColor DarkGray

try {
    $initResponse = Invoke-RestMethod -Method Post -Uri 'http://localhost:8002/api/soliton/init' `
        -ContentType 'application/json' `
        -Body $initPayload
    
    Write-Host "[OK] Soliton init successful!" -ForegroundColor Green
    Write-Host "  Success: $($initResponse.success)" -ForegroundColor White
    Write-Host "  Engine: $($initResponse.engine)" -ForegroundColor White
    Write-Host "  User ID: $($initResponse.user_id)" -ForegroundColor White
    Write-Host "  Message: $($initResponse.message)" -ForegroundColor White
    Write-Host "  Lattice Ready: $($initResponse.lattice_ready)" -ForegroundColor White
} catch {
    Write-Host "[ERROR] Soliton init failed!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    
    # Try to get more details
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $reader.DiscardBufferedData()
        $responseBody = $reader.ReadToEnd()
        Write-Host "`nResponse body:" -ForegroundColor Yellow
        Write-Host $responseBody -ForegroundColor DarkGray
        
        # Check if it's the user_id_id error
        if ($responseBody -match "user_id_id") {
            Write-Host "`n⚠️ FOUND THE TYPO!" -ForegroundColor Red
            Write-Host "The error mentions 'user_id_id' which suggests a typo in the request handler." -ForegroundColor Yellow
        }
    }
}

# Test 4: Test with alternative payload formats
Write-Host "`n[4] Testing alternative payload formats..." -ForegroundColor Yellow

# Try camelCase
$camelCasePayload = @{
    userId = "adminuser"
    latticeReset = $false
} | ConvertTo-Json

Write-Host "`nTrying camelCase format..." -ForegroundColor Gray
try {
    $altResponse = Invoke-RestMethod -Method Post -Uri 'http://localhost:8002/api/soliton/init' `
        -ContentType 'application/json' `
        -Body $camelCasePayload -ErrorAction Stop
    
    Write-Host "[OK] CamelCase format accepted!" -ForegroundColor Green
} catch {
    Write-Host "[INFO] CamelCase format not accepted (expected)" -ForegroundColor Gray
}

# Test 5: Check Soliton stats
Write-Host "`n[5] Checking Soliton stats..." -ForegroundColor Yellow
try {
    $stats = Invoke-RestMethod -Method Get -Uri 'http://localhost:8002/api/soliton/stats/adminuser'
    Write-Host "[OK] Soliton stats retrieved:" -ForegroundColor Green
    Write-Host "  Total Memories: $($stats.totalMemories)" -ForegroundColor White
    Write-Host "  Active Waves: $($stats.activeWaves)" -ForegroundColor White
    Write-Host "  Status: $($stats.status)" -ForegroundColor White
} catch {
    Write-Host "[WARNING] Could not retrieve Soliton stats: $_" -ForegroundColor Yellow
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. If you see the 'user_id_id' error, we need to find where this typo is" -ForegroundColor White
Write-Host "2. Check the API logs for more details about the error" -ForegroundColor White
Write-Host "3. The typo might be in a request validation or middleware layer" -ForegroundColor White
