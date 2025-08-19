# TORI Phase 6-8 Test Script
# PowerShell version of the curl commands

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing TORI Phases 6-8 Fixes" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to test endpoints
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
    Write-Host "  Method: $Method" -ForegroundColor Gray
    
    try {
        $params = @{
            Uri = $Uri
            Method = $Method
            Headers = $Headers
            UseBasicParsing = $true
            TimeoutSec = 5
        }
        
        if ($Body) {
            $params['Body'] = $Body
            $params['ContentType'] = 'application/json'
        }
        
        $response = Invoke-WebRequest @params
        Write-Host "  ✅ Status: $($response.StatusCode)" -ForegroundColor Green
        
        if ($response.Content) {
            $content = $response.Content
            if ($content.Length -gt 100) {
                $content = $content.Substring(0, 100) + "..."
            }
            Write-Host "  Response: $content" -ForegroundColor Gray
        }
        
        return $true
    }
    catch {
        Write-Host "  ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Test different possible ports
$ports = @(5173, 8003, 8002, 8000, 3000)
$workingPort = $null

Write-Host ""
Write-Host "[1/4] Finding active TORI port..." -ForegroundColor Cyan
foreach ($port in $ports) {
    Write-Host "  Checking port $port..." -ForegroundColor Gray -NoNewline
    try {
        $test = Invoke-WebRequest -Uri "http://localhost:$port/api/health" -Method GET -TimeoutSec 2 -UseBasicParsing
        Write-Host " ✅ Active!" -ForegroundColor Green
        $workingPort = $port
        break
    }
    catch {
        Write-Host " ❌ Not responding" -ForegroundColor Red
    }
}

if (-not $workingPort) {
    Write-Host ""
    Write-Host "❌ No TORI instance found running!" -ForegroundColor Red
    Write-Host "💡 Please start TORI first with:" -ForegroundColor Yellow
    Write-Host "   python enhanced_launcher.py --no-browser" -ForegroundColor White
    Write-Host ""
    pause
    exit 1
}

Write-Host ""
Write-Host "✅ Found TORI running on port $workingPort" -ForegroundColor Green
Write-Host ""

# Test Phase 6: Diff Route
Write-Host "[2/4] Testing Phase 6 - ScholarSphere diff route" -ForegroundColor Cyan
$diffTest = Test-Endpoint `
    -Uri "http://localhost:$workingPort/api/concept-mesh/record_diff" `
    -Method "POST" `
    -Body '{"record_id":"smoke_test"}' `
    -Description "POST /api/concept-mesh/record_diff"

# Test Phase 8: Hologram Bridge SSE
Write-Host ""
Write-Host "[3/4] Testing Phase 8 - Hologram bridge SSE" -ForegroundColor Cyan
$holoTest = Test-Endpoint `
    -Uri "http://localhost:$workingPort/holo_renderer/events" `
    -Method "GET" `
    -Description "GET /holo_renderer/events (SSE endpoint)"

# Test available endpoints
Write-Host ""
Write-Host "[4/4] Checking available endpoints" -ForegroundColor Cyan
$healthTest = Test-Endpoint `
    -Uri "http://localhost:$workingPort/api/health" `
    -Method "GET" `
    -Description "GET /api/health"

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$results = @{
    "Phase 6 (Diff Route)" = $diffTest
    "Phase 8 (Hologram SSE)" = $holoTest
    "API Health" = $healthTest
}

$allPassed = $true
foreach ($test in $results.GetEnumerator()) {
    if ($test.Value) {
        Write-Host "✅ $($test.Key)" -ForegroundColor Green
    } else {
        Write-Host "❌ $($test.Key)" -ForegroundColor Red
        $allPassed = $false
    }
}

Write-Host ""
if ($allPassed) {
    Write-Host "🎉 All tests passed!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Some tests failed. Check the logs above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Note: Phase 7 (Oscillator feed) requires triggering concept additions" -ForegroundColor Gray
Write-Host "Watch the logs for 'Published concept_added event' messages" -ForegroundColor Gray

Write-Host ""
pause
