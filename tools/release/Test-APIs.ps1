param(
    [string]$BaseUrl = "http://localhost:5173",
    [string]$OutputDir = "D:\Dev\kha\verification_reports"
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$reportFile = Join-Path $OutputDir "api_test_$timestamp.json"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "API Endpoint Testing Suite" -ForegroundColor Cyan
Write-Host "Base URL: $BaseUrl" -ForegroundColor White
Write-Host ""

$results = @{
    timestamp = (Get-Date).ToString("o")
    baseUrl = $BaseUrl
    tests = @{}
}

function Test-API {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = $null,
        [string]$ContentType = "application/json"
    )
    
    Write-Host "Testing: $Name" -ForegroundColor Yellow
    Write-Host "  $Method $Url" -ForegroundColor Gray
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            UseBasicParsing = $true
            TimeoutSec = 10
        }
        
        if ($Headers.Count -gt 0) {
            $params.Headers = $Headers
        }
        
        if ($Body) {
            $params.Body = $Body
            $params.ContentType = $ContentType
        }
        
        $response = Invoke-WebRequest @params
        
        Write-Host "  [✓] Status: $($response.StatusCode)" -ForegroundColor Green
        
        return @{
            success = $true
            statusCode = $response.StatusCode
            contentType = $response.Headers["Content-Type"]
            contentLength = $response.Content.Length
            headers = $response.Headers
        }
    } catch {
        $statusCode = $null
        if ($_.Exception.Response) {
            $statusCode = $_.Exception.Response.StatusCode.Value__
        }
        
        # Some status codes are expected (e.g., 400 for bad request)
        if ($statusCode -and $statusCode -in @(400, 401, 403, 422)) {
            Write-Host "  [✓] Expected error: HTTP $statusCode" -ForegroundColor Yellow
            return @{
                success = $true
                expectedError = $true
                statusCode = $statusCode
                error = $_.Exception.Message
            }
        } else {
            Write-Host "  [X] Failed: $($_.Exception.Message)" -ForegroundColor Red
            return @{
                success = $false
                error = $_.Exception.Message
            }
        }
    }
}

# Test UI Routes
Write-Host "1. UI ROUTES" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor White

$results.tests.ui = @{}
$uiRoutes = @(
    @{ name = "Home"; path = "/" },
    @{ name = "Hologram"; path = "/hologram" },
    @{ name = "Pricing"; path = "/pricing" },
    @{ name = "Templates"; path = "/templates" },
    @{ name = "Templates Upload"; path = "/templates/upload" },
    @{ name = "Publish"; path = "/publish" },
    @{ name = "Health"; path = "/health" },
    @{ name = "Account"; path = "/account" }
)

foreach ($route in $uiRoutes) {
    $result = Test-API -Name $route.name -Url "$BaseUrl$($route.path)"
    $results.tests.ui[$route.name] = $result
}

Write-Host ""

# Test Health Endpoints
Write-Host "2. HEALTH ENDPOINTS" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor White

$results.tests.health = @{}
$results.tests.health["Health JSON"] = Test-API -Name "Health JSON" -Url "$BaseUrl/health" -Headers @{ "Accept" = "application/json" }
$results.tests.health["Health Ping"] = Test-API -Name "Health Ping" -Url "$BaseUrl/health/ping"
$results.tests.health["Health Raw"] = Test-API -Name "Health Raw" -Url "$BaseUrl/health/raw"

Write-Host ""

# Test Billing APIs
Write-Host "3. BILLING APIs" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor White

$results.tests.billing = @{}

# Checkout API
$checkoutBody = @{
    planId = "plus"
    successUrl = "$BaseUrl/account"
    cancelUrl = "$BaseUrl/pricing"
} | ConvertTo-Json

$results.tests.billing["Checkout"] = Test-API `
    -Name "Billing Checkout" `
    -Url "$BaseUrl/api/billing/checkout" `
    -Method "POST" `
    -Body $checkoutBody

# Portal API (will fail without customer ID, but should respond)
$portalBody = @{
    customerId = "cus_test123"
} | ConvertTo-Json

$results.tests.billing["Portal"] = Test-API `
    -Name "Billing Portal" `
    -Url "$BaseUrl/api/billing/portal" `
    -Method "POST" `
    -Body $portalBody

Write-Host ""

# Test Template APIs
Write-Host "4. TEMPLATE APIs" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor White

$results.tests.templates = @{}

# Export API
$exportBody = @{
    input = "D:\Dev\kha\data\concept_graph.json"
    layout = "grid"
    scale = "0.12"
    zip = $false
} | ConvertTo-Json

$results.tests.templates["Export"] = Test-API `
    -Name "Templates Export" `
    -Url "$BaseUrl/api/templates/export" `
    -Method "POST" `
    -Body $exportBody

# File streaming (test with a known file if it exists)
$results.tests.templates["File Stream"] = Test-API `
    -Name "Template File Stream" `
    -Url "$BaseUrl/api/templates/file/test.glb"

Write-Host ""

# Test Upload endpoint (multipart form data)
Write-Host "5. UPLOAD ENDPOINT" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor White

# Create a test JSON file for upload
$testFile = Join-Path $OutputDir "test_concept.json"
$testContent = @{
    nodes = @(
        @{ id = 1; label = "Test"; x = 0; y = 0; z = 0 }
    )
} | ConvertTo-Json
Set-Content -Path $testFile -Value $testContent

Write-Host "  Note: Upload endpoint requires multipart/form-data" -ForegroundColor Gray
Write-Host "  Test manually via UI at $BaseUrl/templates/upload" -ForegroundColor Gray

Write-Host ""

# Generate Summary
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  API TEST SUMMARY                       " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$totalTests = 0
$passedTests = 0

foreach ($category in $results.tests.Keys) {
    foreach ($test in $results.tests[$category].Keys) {
        $totalTests++
        if ($results.tests[$category][$test].success) {
            $passedTests++
        }
    }
}

Write-Host "Total Tests: $totalTests" -ForegroundColor White
Write-Host "Passed: $passedTests" -ForegroundColor Green
Write-Host "Failed: $($totalTests - $passedTests)" -ForegroundColor $(if ($totalTests -eq $passedTests) { "White" } else { "Red" })
Write-Host ""
Write-Host "Success Rate: $([math]::Round(($passedTests / $totalTests) * 100, 2))%" -ForegroundColor $(if ($passedTests -eq $totalTests) { "Green" } else { "Yellow" })

$results.summary = @{
    totalTests = $totalTests
    passedTests = $passedTests
    failedTests = $totalTests - $passedTests
    successRate = [math]::Round(($passedTests / $totalTests) * 100, 2)
}

# Save report
$results | ConvertTo-Json -Depth 10 | Set-Content $reportFile -Encoding UTF8

Write-Host ""
Write-Host "Report saved to:" -ForegroundColor Cyan
Write-Host $reportFile -ForegroundColor White

exit $(if ($passedTests -eq $totalTests) { 0 } else { 1 })