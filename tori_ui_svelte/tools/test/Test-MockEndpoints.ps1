# Test-MockEndpoints.ps1
# Quick test to verify mock endpoints are working

param(
    [int]$Port = 3000
)

Write-Host "Testing mock endpoints on port $Port..." -ForegroundColor Cyan
Write-Host "Current IRIS_USE_MOCKS: $($env:IRIS_USE_MOCKS)" -ForegroundColor Yellow

$endpoints = @(
    "/api/pdf/stats",
    "/api/memory/state"
)

foreach ($endpoint in $endpoints) {
    $url = "http://localhost:$Port$endpoint"
    Write-Host "`nTesting: $url" -ForegroundColor Yellow
    
    try {
        $response = Invoke-RestMethod -Uri $url -Method Get
        
        if ($response.ok -eq $true) {
            Write-Host "  ✓ SUCCESS - Mock response received" -ForegroundColor Green
            Write-Host "    Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
        } else {
            Write-Host "  ✗ FAIL - Response not OK" -ForegroundColor Red
            Write-Host "    Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ✗ ERROR - $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`nTip: Make sure IRIS_USE_MOCKS=1 is set in .env.local or environment" -ForegroundColor Cyan
