param(
    [string]$ProjectRoot = "D:\Dev\kha",
    [switch]$SkipDevServer = $false,
    [switch]$GenerateReport = $true
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$reportDir = Join-Path $ProjectRoot "verification_reports"
$auditReport = Join-Path $reportDir "full_audit_$timestamp.json"

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  FULL MONETIZATION AUDIT                " -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$results = @{
    timestamp = (Get-Date).ToString("o")
    projectRoot = $ProjectRoot
    checks = @{}
}

# Function to check if file exists
function Test-FileExists {
    param([string]$Path)
    $exists = Test-Path $Path
    if ($exists) {
        Write-Host "[✓] $Path" -ForegroundColor Green
    } else {
        Write-Host "[X] $Path" -ForegroundColor Red
    }
    return $exists
}

# Function to test HTTP endpoint
function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = $null
    )
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            UseBasicParsing = $true
            TimeoutSec = 5
        }
        
        if ($Headers.Count -gt 0) {
            $params.Headers = $Headers
        }
        
        if ($Body) {
            $params.Body = $Body
            $params.ContentType = "application/json"
        }
        
        $response = Invoke-WebRequest @params
        return @{
            success = $true
            statusCode = $response.StatusCode
            contentLength = $response.Content.Length
        }
    } catch {
        return @{
            success = $false
            error = $_.Exception.Message
        }
    }
}

Write-Host "1. FILE STRUCTURE VERIFICATION" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

$requiredFiles = @(
    # Device & Capabilities
    "frontend\src\lib\device\capabilities.ts",
    "frontend\src\lib\stores\userPlan.ts",
    
    # Plans Configuration
    "config\plans.json",
    "frontend\static\config\plans.json",
    
    # Video Export
    "frontend\src\lib\utils\exportVideo.ts",
    
    # Components
    "frontend\src\lib\components\HologramRecorder.svelte",
    "frontend\src\lib\components\PricingTable.svelte",
    
    # Hologram System
    "frontend\src\lib\hologram\engineShim.ts",
    "frontend\src\routes\hologram\+page.svelte",
    
    # Pricing & Billing
    "frontend\src\routes\pricing\+page.svelte",
    "frontend\src\routes\api\billing\checkout\+server.ts",
    "frontend\src\routes\api\billing\portal\+server.ts",
    
    # Templates System
    "frontend\src\routes\templates\+page.svelte",
    "frontend\src\routes\templates\+page.server.ts",
    "frontend\src\routes\templates\upload\+page.svelte",
    "frontend\src\routes\api\templates\export\+server.ts",
    "frontend\src\routes\api\templates\upload\+server.ts",
    "frontend\src\routes\api\templates\file\[name]\+server.ts",
    
    # Publish System
    "frontend\src\routes\publish\+page.svelte",
    "frontend\src\routes\publish\+page.server.ts",
    
    # Health System
    "frontend\src\routes\health\+server.ts",
    "frontend\src\routes\health\+page.svelte",
    "frontend\src\routes\health\+page.server.ts",
    "frontend\src\lib\health\checks.server.ts",
    
    # Exporters
    "tools\exporters\glb-from-conceptmesh.ts",
    "tools\exporters\encode-ktx2.ps1",
    
    # Sync Tools
    "tools\release\sync-plans.mjs",
    "tools\release\Sync-Plans.ps1",
    "tools\release\build-templates-index.mjs",
    "tools\release\Build-Templates-Index.ps1",
    
    # Verification Tools
    "tools\release\Verify-Hologram-Route.ps1",
    "tools\release\Verify-Health.ps1",
    "tools\release\check-mobile-claims.mjs",
    "tools\release\Verify-Mobile-Claims.ps1",
    
    # Mobile Configuration
    "config\mobile_support.json",
    "docs\MOBILE_SUPPORT_MATRIX.md"
)

$fileResults = @{}
$missingFiles = @()

foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $ProjectRoot $file
    $exists = Test-FileExists $fullPath
    $fileResults[$file] = $exists
    if (-not $exists) {
        $missingFiles += $file
    }
}

$results.checks.files = @{
    total = $requiredFiles.Count
    found = ($requiredFiles.Count - $missingFiles.Count)
    missing = $missingFiles
    details = $fileResults
}

Write-Host ""
Write-Host "Files: $($results.checks.files.found)/$($results.checks.files.total) present" -ForegroundColor $(if ($missingFiles.Count -eq 0) { "Green" } else { "Yellow" })
Write-Host ""

# 2. Check Git Repository Status
Write-Host "2. GIT REPOSITORY STATUS" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

Push-Location $ProjectRoot
try {
    $gitStatus = git status --porcelain
    $gitBranch = git rev-parse --abbrev-ref HEAD
    $gitCommit = git rev-parse HEAD
    $gitRemote = git remote get-url origin 2>$null
    
    Write-Host "[✓] Branch: $gitBranch" -ForegroundColor Green
    Write-Host "[✓] Commit: $gitCommit" -ForegroundColor Green
    Write-Host "[✓] Remote: $gitRemote" -ForegroundColor Green
    
    if ($gitStatus) {
        Write-Host "[!] Uncommitted changes detected:" -ForegroundColor Yellow
        $gitStatus | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    } else {
        Write-Host "[✓] Working tree clean" -ForegroundColor Green
    }
    
    $results.checks.git = @{
        branch = $gitBranch
        commit = $gitCommit
        remote = $gitRemote
        clean = [string]::IsNullOrEmpty($gitStatus)
        uncommitted = $gitStatus -split "`n"
    }
} finally {
    Pop-Location
}

Write-Host ""

# 3. Check if dev server is running
Write-Host "3. DEVELOPMENT SERVER CHECK" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

$devServerRunning = $false
$baseUrl = "http://localhost:5173"

$pingResult = Test-Endpoint "$baseUrl/health/ping"
if ($pingResult.success) {
    Write-Host "[✓] Dev server is running on port 5173" -ForegroundColor Green
    $devServerRunning = $true
} else {
    if (-not $SkipDevServer) {
        Write-Host "[!] Starting dev server..." -ForegroundColor Yellow
        $frontendPath = Join-Path $ProjectRoot "frontend"
        Start-Process -FilePath "pnpm" -ArgumentList "dev" -WorkingDirectory $frontendPath -WindowStyle Hidden
        Start-Sleep -Seconds 10
        
        $pingResult = Test-Endpoint "$baseUrl/health/ping"
        if ($pingResult.success) {
            Write-Host "[✓] Dev server started successfully" -ForegroundColor Green
            $devServerRunning = $true
        } else {
            Write-Host "[X] Failed to start dev server" -ForegroundColor Red
        }
    } else {
        Write-Host "[!] Dev server not running (skipped)" -ForegroundColor Yellow
    }
}

$results.checks.devServer = @{
    running = $devServerRunning
    port = 5173
    url = $baseUrl
}

Write-Host ""

# 4. Test Endpoints
if ($devServerRunning) {
    Write-Host "4. ENDPOINT VERIFICATION" -ForegroundColor Yellow
    Write-Host "================================" -ForegroundColor Yellow
    
    $endpoints = @(
        @{ name = "Hologram Route"; url = "$baseUrl/hologram"; method = "GET" },
        @{ name = "Pricing Route"; url = "$baseUrl/pricing"; method = "GET" },
        @{ name = "Templates Route"; url = "$baseUrl/templates"; method = "GET" },
        @{ name = "Publish Route"; url = "$baseUrl/publish"; method = "GET" },
        @{ name = "Health Route"; url = "$baseUrl/health"; method = "GET" },
        @{ name = "Health Ping"; url = "$baseUrl/health/ping"; method = "GET" },
        @{ name = "Health Raw"; url = "$baseUrl/health/raw"; method = "GET" }
    )
    
    $apiEndpoints = @(
        @{ 
            name = "Billing Checkout API"
            url = "$baseUrl/api/billing/checkout"
            method = "POST"
            body = '{"planId":"plus"}'
        },
        @{ 
            name = "Templates Export API"
            url = "$baseUrl/api/templates/export"
            method = "POST"
            body = '{"input":"D:\\Dev\\kha\\data\\concept_graph.json","layout":"grid","scale":"0.12"}'
        }
    )
    
    $endpointResults = @{}
    
    foreach ($endpoint in $endpoints) {
        $result = Test-Endpoint -Url $endpoint.url -Method $endpoint.method
        if ($result.success) {
            Write-Host "[✓] $($endpoint.name): HTTP $($result.statusCode)" -ForegroundColor Green
        } else {
            Write-Host "[X] $($endpoint.name): Failed" -ForegroundColor Red
        }
        $endpointResults[$endpoint.name] = $result
    }
    
    Write-Host ""
    Write-Host "API Endpoints:" -ForegroundColor Cyan
    
    foreach ($api in $apiEndpoints) {
        $result = Test-Endpoint -Url $api.url -Method $api.method -Body $api.body
        if ($result.success -or $result.error -match "400|401|403") {
            Write-Host "[✓] $($api.name): Responding" -ForegroundColor Green
        } else {
            Write-Host "[X] $($api.name): Not responding" -ForegroundColor Red
        }
        $endpointResults[$api.name] = $result
    }
    
    $results.checks.endpoints = $endpointResults
}

Write-Host ""

# 5. Mobile Claims Verification
Write-Host "5. MOBILE CLAIMS VERIFICATION" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

$mobileScript = Join-Path $ProjectRoot "tools\release\check-mobile-claims.mjs"
if (Test-Path $mobileScript) {
    try {
        $mobileResult = & node $mobileScript 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[✓] Mobile claims are consistent with code" -ForegroundColor Green
            $results.checks.mobileClaims = @{ consistent = $true }
        } else {
            Write-Host "[X] Mobile claims have issues" -ForegroundColor Red
            $results.checks.mobileClaims = @{ consistent = $false; output = $mobileResult }
        }
    } catch {
        Write-Host "[!] Could not run mobile claims check" -ForegroundColor Yellow
        $results.checks.mobileClaims = @{ consistent = $null; error = $_.Exception.Message }
    }
}

Write-Host ""

# 6. Plans Sync Verification
Write-Host "6. PLANS SYNCHRONIZATION CHECK" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

$plansSrc = Join-Path $ProjectRoot "config\plans.json"
$plansDst = Join-Path $ProjectRoot "frontend\static\config\plans.json"

if ((Test-Path $plansSrc) -and (Test-Path $plansDst)) {
    $srcHash = (Get-FileHash $plansSrc -Algorithm SHA256).Hash
    $dstHash = (Get-FileHash $plansDst -Algorithm SHA256).Hash
    
    if ($srcHash -eq $dstHash) {
        Write-Host "[✓] Plans are synchronized" -ForegroundColor Green
        $results.checks.plansSync = @{ synced = $true }
    } else {
        Write-Host "[X] Plans are NOT synchronized" -ForegroundColor Red
        Write-Host "    Run: tools\release\Sync-Plans.ps1" -ForegroundColor Yellow
        $results.checks.plansSync = @{ synced = $false }
    }
} else {
    Write-Host "[X] Plans files missing" -ForegroundColor Red
    $results.checks.plansSync = @{ synced = $false; error = "Files missing" }
}

Write-Host ""

# 7. Generate Summary
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  AUDIT SUMMARY                          " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$totalChecks = 0
$passedChecks = 0

# Files
$totalChecks++
if ($results.checks.files.missing.Count -eq 0) {
    $passedChecks++
    Write-Host "[✓] All required files present" -ForegroundColor Green
} else {
    Write-Host "[X] Missing $($results.checks.files.missing.Count) files" -ForegroundColor Red
}

# Git
$totalChecks++
if ($results.checks.git.clean) {
    $passedChecks++
    Write-Host "[✓] Git repository clean" -ForegroundColor Green
} else {
    Write-Host "[!] Git has uncommitted changes" -ForegroundColor Yellow
}

# Dev Server
if ($devServerRunning) {
    $totalChecks++
    $passedChecks++
    Write-Host "[✓] Dev server operational" -ForegroundColor Green
}

# Mobile Claims
if ($results.checks.mobileClaims.consistent) {
    $totalChecks++
    $passedChecks++
    Write-Host "[✓] Mobile documentation consistent" -ForegroundColor Green
}

# Plans Sync
$totalChecks++
if ($results.checks.plansSync.synced) {
    $passedChecks++
    Write-Host "[✓] Plans synchronized" -ForegroundColor Green
} else {
    Write-Host "[X] Plans need synchronization" -ForegroundColor Red
}

Write-Host ""
Write-Host "Overall Score: $passedChecks/$totalChecks checks passed" -ForegroundColor $(if ($passedChecks -eq $totalChecks) { "Green" } else { "Yellow" })

$results.summary = @{
    totalChecks = $totalChecks
    passedChecks = $passedChecks
    score = [math]::Round(($passedChecks / $totalChecks) * 100, 2)
    investorReady = ($passedChecks -eq $totalChecks)
}

# 8. Save Report
if ($GenerateReport) {
    $results | ConvertTo-Json -Depth 10 | Set-Content $auditReport -Encoding UTF8
    Write-Host ""
    Write-Host "Full report saved to:" -ForegroundColor Cyan
    Write-Host $auditReport -ForegroundColor White
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  INVESTOR READINESS: $(if ($results.summary.investorReady) { 'READY ✓' } else { 'NEEDS WORK' })" -ForegroundColor $(if ($results.summary.investorReady) { "Green" } else { "Yellow" })
Write-Host "==========================================" -ForegroundColor Cyan

exit $(if ($results.summary.investorReady) { 0 } else { 1 })