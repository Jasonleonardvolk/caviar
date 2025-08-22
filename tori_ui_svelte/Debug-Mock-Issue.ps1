# Debug-Mock-Issue.ps1
# Diagnostic script to figure out why mocks aren't working

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "       Debugging Mock Endpoint Issue           " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check environment files
Write-Host "[1] Checking environment files..." -ForegroundColor Yellow

$envFiles = @(".env", ".env.local", ".env.production")
foreach ($file in $envFiles) {
    if (Test-Path $file) {
        Write-Host "  Found $file`:" -ForegroundColor Green
        $content = Get-Content $file
        $content | Where-Object { $_ -match "IRIS_USE_MOCKS" } | ForEach-Object {
            Write-Host "    $_" -ForegroundColor Cyan
        }
    } else {
        Write-Host "  $file not found" -ForegroundColor Gray
    }
}

# Check current environment
Write-Host "`n[2] Current PowerShell environment..." -ForegroundColor Yellow
Write-Host "  IRIS_USE_MOCKS = $env:IRIS_USE_MOCKS" -ForegroundColor Cyan
Write-Host "  PORT = $env:PORT" -ForegroundColor Cyan
Write-Host "  NODE_ENV = $env:NODE_ENV" -ForegroundColor Cyan

# Check if compiled output exists
Write-Host "`n[3] Checking compiled output..." -ForegroundColor Yellow

$compiledPdf = ".\.svelte-kit\output\server\entries\endpoints\api\pdf\stats\_server.ts.js"
$compiledMemory = ".\.svelte-kit\output\server\entries\endpoints\api\memory\state\_server.ts.js"

if (Test-Path $compiledPdf) {
    Write-Host "  PDF endpoint compiled file exists" -ForegroundColor Green
    $content = Get-Content $compiledPdf -Raw
    
    # Check for environment variable usage
    if ($content -match 'IRIS_USE_MOCKS') {
        Write-Host "    Contains IRIS_USE_MOCKS check ✓" -ForegroundColor Green
        
        # Extract the relevant lines
        $lines = $content -split "`n"
        $mockLines = $lines | Select-String -Pattern "IRIS_USE_MOCKS|note.*mock" -Context 2,2
        Write-Host "    Relevant lines:" -ForegroundColor Gray
        $mockLines | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
    } else {
        Write-Host "    NO IRIS_USE_MOCKS check found!" -ForegroundColor Red
        Write-Host "    First 30 lines:" -ForegroundColor Yellow
        $lines = $content -split "`n" | Select-Object -First 30
        $lines | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
    }
} else {
    Write-Host "  PDF endpoint NOT compiled!" -ForegroundColor Red
}

# Test with curl-like request showing headers
Write-Host "`n[4] Testing endpoints with detailed output..." -ForegroundColor Yellow

$testUrl = "http://127.0.0.1:3000/api/pdf/stats"

try {
    # First check if server is running
    $tcp = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
    if (-not $tcp) {
        Write-Host "  No server listening on port 3000!" -ForegroundColor Red
        Write-Host "  Start the server first with:" -ForegroundColor Yellow
        Write-Host "    .\Force-Fresh-Build.ps1" -ForegroundColor White
    } else {
        Write-Host "  Server is listening on port 3000" -ForegroundColor Green
        
        # Make request with verbose output
        Write-Host "`n  Making request to $testUrl..." -ForegroundColor Gray
        $response = Invoke-WebRequest -Uri $testUrl -UseBasicParsing -Method GET
        
        Write-Host "  Status: $($response.StatusCode)" -ForegroundColor Cyan
        Write-Host "  Headers:" -ForegroundColor Gray
        $response.Headers.GetEnumerator() | ForEach-Object {
            Write-Host "    $($_.Key): $($_.Value)" -ForegroundColor DarkGray
        }
        
        Write-Host "  Body:" -ForegroundColor Gray
        $body = $response.Content | ConvertFrom-Json
        Write-Host "    $($body | ConvertTo-Json -Compress)" -ForegroundColor Cyan
        
        if ($body.note -eq 'mock') {
            Write-Host "`n  SUCCESS: Mock is working!" -ForegroundColor Green
        } elseif ($body.error -like "*not configured*") {
            Write-Host "`n  PROBLEM: Still returning 'not configured'" -ForegroundColor Red
            Write-Host "  This means the environment variable isn't being read" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "  ERROR: $_" -ForegroundColor Red
}

# Check build output folder structure
Write-Host "`n[5] Checking build folder..." -ForegroundColor Yellow

if (Test-Path "build") {
    Write-Host "  Build folder exists" -ForegroundColor Green
    
    # Check if there's a .env file in build
    if (Test-Path "build\.env") {
        Write-Host "  Found build\.env:" -ForegroundColor Yellow
        Get-Content "build\.env" | ForEach-Object {
            Write-Host "    $_" -ForegroundColor DarkGray
        }
    }
    
    # Check package.json in build
    if (Test-Path "build\package.json") {
        Write-Host "  Build has package.json" -ForegroundColor Green
    }
} else {
    Write-Host "  Build folder doesn't exist!" -ForegroundColor Red
}

# Suggestions
Write-Host "`n[6] Diagnostic Summary..." -ForegroundColor Yellow

$issues = @()

if ($env:IRIS_USE_MOCKS -ne "1") {
    $issues += "IRIS_USE_MOCKS not set in PowerShell environment"
}

if (-not (Test-Path ".env.local") -or -not ((Get-Content ".env.local" -ErrorAction SilentlyContinue) -match "IRIS_USE_MOCKS=1")) {
    $issues += ".env.local missing or doesn't have IRIS_USE_MOCKS=1"
}

if (-not (Test-Path $compiledPdf)) {
    $issues += "Compiled output doesn't exist"
} elseif (-not ((Get-Content $compiledPdf -Raw) -match "IRIS_USE_MOCKS")) {
    $issues += "Compiled output doesn't check IRIS_USE_MOCKS"
}

if ($issues.Count -gt 0) {
    Write-Host "`n  Issues found:" -ForegroundColor Red
    $issues | ForEach-Object {
        Write-Host "    - $_" -ForegroundColor Yellow
    }
    
    Write-Host "`n  Recommended fix:" -ForegroundColor Cyan
    Write-Host "    1. Set environment:" -ForegroundColor Gray
    Write-Host '       $env:IRIS_USE_MOCKS = "1"' -ForegroundColor White
    Write-Host "    2. Clear cache and rebuild:" -ForegroundColor Gray
    Write-Host "       Remove-Item .\.svelte-kit -Recurse -Force" -ForegroundColor White
    Write-Host "       pnpm run build" -ForegroundColor White
    Write-Host "    3. Start with environment:" -ForegroundColor Gray
    Write-Host '       $env:IRIS_USE_MOCKS = "1"; node build/index.js' -ForegroundColor White
} else {
    Write-Host "`n  Everything looks correct!" -ForegroundColor Green
    Write-Host "  If mocks still don't work, try:" -ForegroundColor Yellow
    Write-Host "    .\Force-Fresh-Build.ps1" -ForegroundColor White
}

Write-Host "`nDone." -ForegroundColor Green
