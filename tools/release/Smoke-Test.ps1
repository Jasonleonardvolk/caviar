Param(
    [string]$BaseUrl = "http://localhost:4173"
)

Write-Host "Running smoke tests against $BaseUrl" -ForegroundColor Cyan

# Helper to test an endpoint
function Test-Endpoint($Path) {
    $url = "$BaseUrl$Path"
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
        if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 400) {
            Write-Host "[OK] $url → $($resp.StatusCode)" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] $url → $($resp.StatusCode)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "[ERROR] $url → $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Pages
Test-Endpoint "/"
Test-Endpoint "/health"
Test-Endpoint "/upload"
Test-Endpoint "/vault"
Test-Endpoint "/ghost-history"
Test-Endpoint "/elfin"

# Sample API endpoints (if running with adapter-node SSR build)
Test-Endpoint "/api/list"
Test-Endpoint "/api/pdf/stats"
Test-Endpoint "/api/memory/state"

Write-Host "Smoke test complete." -ForegroundColor Cyan
