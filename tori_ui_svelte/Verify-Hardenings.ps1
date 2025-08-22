# Verify-Hardenings.ps1
# Verifies all hardenings from the review are properly in place

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "       Verifying All Hardenings In Place       " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check 1: Verify source files have mock gates
Write-Host "[1] Checking source files for mock gates..." -ForegroundColor Yellow

$pdfSource = "src\routes\api\pdf\stats\+server.ts"
$memorySource = "src\routes\api\memory\state\+server.ts"

if (Test-Path $pdfSource) {
    $content = Get-Content $pdfSource -Raw
    if ($content -match "env\.IRIS_USE_MOCKS === '1'") {
        Write-Host "  PDF stats: Has mock gate" -ForegroundColor Green
    } else {
        Write-Host "  PDF stats: MISSING mock gate!" -ForegroundColor Red
        $allGood = $false
    }
} else {
    Write-Host "  PDF stats: FILE NOT FOUND!" -ForegroundColor Red
    $allGood = $false
}

if (Test-Path $memorySource) {
    $content = Get-Content $memorySource -Raw
    if ($content -match "env\.IRIS_USE_MOCKS === '1'") {
        Write-Host "  Memory state: Has mock gate" -ForegroundColor Green
    } else {
        Write-Host "  Memory state: MISSING mock gate!" -ForegroundColor Red
        $allGood = $false
    }
} else {
    Write-Host "  Memory state: FILE NOT FOUND!" -ForegroundColor Red
    $allGood = $false
}

# Check 2: Verify renderer SSR guard exists
Write-Host "`n[2] Checking renderer SSR guard..." -ForegroundColor Yellow

$rendererSSR = "src\routes\renderer\+page.server.ts"
if (Test-Path $rendererSSR) {
    $content = Get-Content $rendererSSR -Raw
    if ($content -match "env\.IRIS_USE_MOCKS === '1'") {
        Write-Host "  Renderer SSR: Has mock guard" -ForegroundColor Green
    } else {
        Write-Host "  Renderer SSR: MISSING mock guard!" -ForegroundColor Red
        $allGood = $false
    }
} else {
    Write-Host "  Renderer SSR: FILE NOT FOUND!" -ForegroundColor Red
    $allGood = $false
}

# Check 3: Verify Reset-And-Ship.ps1 has PM2 --update-env
Write-Host "`n[3] Checking Reset-And-Ship.ps1 for PM2 fixes..." -ForegroundColor Yellow

$resetShip = "tools\release\Reset-And-Ship.ps1"
if (Test-Path $resetShip) {
    $content = Get-Content $resetShip -Raw
    
    if ($content -match "--update-env") {
        Write-Host "  PM2 --update-env: Found" -ForegroundColor Green
    } else {
        Write-Host "  PM2 --update-env: MISSING!" -ForegroundColor Red
        $allGood = $false
    }
    
    if ($content -match "function Test-Url") {
        Write-Host "  Clean self-test block: Found" -ForegroundColor Green
    } else {
        Write-Host "  Clean self-test block: MISSING!" -ForegroundColor Red
        $allGood = $false
    }
} else {
    Write-Host "  Reset-And-Ship.ps1: FILE NOT FOUND!" -ForegroundColor Red
    $allGood = $false
}

# Check 4: Verify compiled output (if exists)
Write-Host "`n[4] Checking compiled output for mocks..." -ForegroundColor Yellow

$compiledPdf = ".\.svelte-kit\output\server\entries\endpoints\api\pdf\stats\_server.ts.js"
$compiledMemory = ".\.svelte-kit\output\server\entries\endpoints\api\memory\state\_server.ts.js"

if ((Test-Path $compiledPdf) -and (Test-Path $compiledMemory)) {
    $pdfCompiled = Get-Content $compiledPdf -Raw
    $memoryCompiled = Get-Content $compiledMemory -Raw
    
    if ($pdfCompiled -match 'note.*mock' -and $memoryCompiled -match 'note.*mock') {
        Write-Host "  Compiled output: Contains mocks" -ForegroundColor Green
    } else {
        Write-Host "  Compiled output: Mocks NOT FOUND (rebuild needed)" -ForegroundColor Yellow
        Write-Host "  Run: pnpm run build" -ForegroundColor Gray
    }
} else {
    Write-Host "  Compiled output: Not built yet" -ForegroundColor Gray
    Write-Host "  This is OK - will be built during deployment" -ForegroundColor Gray
}

# Check 5: Verify environment file
Write-Host "`n[5] Checking environment configuration..." -ForegroundColor Yellow

if (Test-Path ".env.local") {
    $content = Get-Content ".env.local" -Raw
    if ($content -match "IRIS_USE_MOCKS=1") {
        Write-Host "  .env.local: Has IRIS_USE_MOCKS=1" -ForegroundColor Green
    } else {
        Write-Host "  .env.local: Missing IRIS_USE_MOCKS=1" -ForegroundColor Yellow
    }
} else {
    Write-Host "  .env.local: FILE NOT FOUND" -ForegroundColor Yellow
    Write-Host "  This is OK - env vars can be set at runtime" -ForegroundColor Gray
}

# Check 6: Verify upload directory
Write-Host "`n[6] Checking upload directory..." -ForegroundColor Yellow

if (Test-Path "var\uploads") {
    Write-Host "  var\uploads: Exists" -ForegroundColor Green
} else {
    Write-Host "  var\uploads: Missing (will be created)" -ForegroundColor Gray
}

# Summary
Write-Host "`n================================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "        ALL HARDENINGS IN PLACE!               " -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Ready to run:" -ForegroundColor Yellow
    Write-Host "  .\Final-Runbook-Clean.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run individual commands:" -ForegroundColor Yellow
    Write-Host "  .\Bulletproof-Build-And-Ship.ps1 -Mode mock -UsePM2" -ForegroundColor White
} else {
    Write-Host "        SOME HARDENINGS MISSING!               " -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Fix missing items above, then re-run this verification." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Quick fix all:" -ForegroundColor Yellow
    Write-Host "  .\Apply-All-Fixes.ps1" -ForegroundColor White
}
Write-Host ""
