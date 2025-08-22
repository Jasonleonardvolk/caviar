#!/usr/bin/env pwsh
# Test-ARAssets.ps1
# Verify AR Assets system is properly configured

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "     AR Assets System Test" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$tests = @()
$projectRoot = "D:\Dev\kha\tori_ui_svelte"

# Test 1: Directory structure
$dirs = @(
    "assets\3d\luxury",
    "assets\3d\luxury\sneakers\free\cgtrader_generic_sneaker\raw",
    "assets\3d\luxury\sneakers\free\cgtrader_generic_sneaker\opt",
    "assets\3d\luxury\watches\free\sketchfab_lowpoly_watch\raw",
    "assets\3d\luxury\watches\free\sketchfab_lowpoly_watch\opt",
    "assets\3d\luxury\perfume\free\sketchfab_perfume_bottle\raw",
    "assets\3d\luxury\perfume\free\sketchfab_perfume_bottle\opt"
)

foreach ($dir in $dirs) {
    $path = Join-Path $projectRoot $dir
    if (Test-Path $path) {
        $tests += @{Name="Directory: $dir"; Status="PASS"}
    } else {
        $tests += @{Name="Directory: $dir"; Status="FAIL"}
    }
}

# Test 2: Core files
$files = @(
    "assets\3d\luxury\ASSET_MANIFEST.json",
    "assets\3d\luxury\ASSET_LICENSES.md",
    "tools\assets\ingest-3d.ps1",
    "tools\assets\optimize-gltf.mjs",
    "tools\assets\validate-manifest.mjs",
    "src\routes\assets\+page.svelte",
    "src\routes\api\assets\+server.ts"
)

foreach ($file in $files) {
    $path = Join-Path $projectRoot $file
    if (Test-Path $path) {
        $tests += @{Name="File: $(Split-Path $file -Leaf)"; Status="PASS"}
    } else {
        $tests += @{Name="File: $(Split-Path $file -Leaf)"; Status="FAIL"}
    }
}

# Test 3: Validate manifest
Write-Host "Validating manifest..." -ForegroundColor Yellow
Set-Location $projectRoot
$manifestTest = node tools\assets\validate-manifest.mjs 2>&1
if ($LASTEXITCODE -eq 0) {
    $tests += @{Name="Manifest validation"; Status="PASS"}
} else {
    $tests += @{Name="Manifest validation"; Status="WARN"}
}

# Display results
Write-Host "`nTest Results:" -ForegroundColor Yellow
Write-Host "-------------" -ForegroundColor Gray

$passCount = 0
$failCount = 0
$warnCount = 0

foreach ($test in $tests) {
    $color = switch ($test.Status) {
        "PASS" { $passCount++; "Green" }
        "FAIL" { $failCount++; "Red" }
        "WARN" { $warnCount++; "Yellow" }
    }
    
    $symbol = switch ($test.Status) {
        "PASS" { "[OK]" }
        "FAIL" { "[X]" }
        "WARN" { "[!]" }
    }
    
    Write-Host "$symbol $($test.Name)" -ForegroundColor $color
}

# Summary
Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  Passed: $passCount" -ForegroundColor Green
if ($warnCount -gt 0) {
    Write-Host "  Warnings: $warnCount" -ForegroundColor Yellow
}
if ($failCount -gt 0) {
    Write-Host "  Failed: $failCount" -ForegroundColor Red
}

# Overall status
Write-Host "`nOverall Status: " -NoNewline
if ($failCount -eq 0) {
    Write-Host "AR ASSETS READY! [OK]" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. Download free 3D models (see AR_ASSETS_GUIDE.md)" -ForegroundColor White
    Write-Host "  2. Run: pwsh tools\assets\ingest-3d.ps1 to optimize" -ForegroundColor White
    Write-Host "  3. View at: http://localhost:3000/assets" -ForegroundColor White
} else {
    Write-Host "SETUP INCOMPLETE [X]" -ForegroundColor Red
    Write-Host "`nRun: .\setup-ar-assets.bat to complete setup" -ForegroundColor Yellow
}

Write-Host ""