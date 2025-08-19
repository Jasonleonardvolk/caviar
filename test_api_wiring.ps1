# Test API Endpoints After Wiring Fixes
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing API Wiring Fixes (Phases 6-8)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to test endpoint
function Test-Endpoint {
    param(
        [string]$Uri,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = "",
        [string]$Description = ""
    )
    
    Write-Host "[TEST] $Description" -ForegroundColor Yellow
    Write-Host "  URI: $Uri" -ForegroundColor Gray
    
    try {
        $params = @{
            Uri = $Uri
            Method = $Method
            Headers = $Headers
            UseBasicParsing = $true
            TimeoutSec = 5
        }
        
        if ($Body -and $Method -eq "POST") {
            $params['Body'] = $Body
            $params['ContentType'] = 'application/json'
        }
        
        $response = Invoke-WebRequest @params
        Write-Host "  ✅ Status: $($response.StatusCode)" -ForegroundColor Green
        
        if ($response.Content -and $response.Content.Length -lt 200) {
            Write-Host "  Response: $($response.Content)" -ForegroundColor Gray
        }
        
        return $true
    }
    catch {
        Write-Host "  ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Test the main API on port 9000
Write-Host "[1/5] Testing main API health" -ForegroundColor Cyan
$apiHealth = Test-Endpoint -Uri "http://localhost:9000/api/health" -Description "Main API Health"

# Test Phase 6 - Diff route
Write-Host ""
Write-Host "[2/5] Testing Phase 6 - Diff route" -ForegroundColor Cyan
$diffTest = Test-Endpoint `
    -Uri "http://localhost:9000/api/concept-mesh/record_diff" `
    -Method "POST" `
    -Body '{"record_id":"test"}' `
    -Description "POST /api/concept-mesh/record_diff"

# Test Phase 8 - Hologram SSE
Write-Host ""
Write-Host "[3/5] Testing Phase 8 - Hologram SSE" -ForegroundColor Cyan
$holoTest = Test-Endpoint `
    -Uri "http://localhost:9000/holo_renderer/events" `
    -Description "GET /holo_renderer/events"

# Test MCP server if running
Write-Host ""
Write-Host "[4/5] Testing MCP server" -ForegroundColor Cyan
$mcpTest = Test-Endpoint `
    -Uri "http://localhost:8100/api/system/status" `
    -Description "MCP System Status"

# Test Prajna API if running
Write-Host ""
Write-Host "[5/5] Testing Prajna API" -ForegroundColor Cyan
$prajnaTest = Test-Endpoint `
    -Uri "http://localhost:8002/api/health" `
    -Description "Prajna API Health"

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$results = @{
    "Main API (9000)" = $apiHealth
    "Phase 6 Diff Route" = $diffTest
    "Phase 8 Hologram SSE" = $holoTest
    "MCP Server (8100)" = $mcpTest
    "Prajna API (8002)" = $prajnaTest
}

$passed = 0
$total = 0
foreach ($test in $results.GetEnumerator()) {
    $total++
    if ($test.Value) {
        Write-Host "✅ $($test.Key)" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "❌ $($test.Key)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Passed: $passed/$total" -ForegroundColor $(if ($passed -eq $total) { "Green" } else { "Yellow" })

if ($passed -lt 3) {
    Write-Host ""
    Write-Host "To start the main API:" -ForegroundColor Yellow
    Write-Host "  uvicorn api.main:app --port 9000" -ForegroundColor White
}

Write-Host ""
pause
