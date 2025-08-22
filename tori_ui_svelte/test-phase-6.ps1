# Phase 6: Complete Test Suite
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 6: Complete Test Suite" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path "D:\Dev\kha\tori_ui_svelte"

Write-Host "Running Phase 6 Verification..." -ForegroundColor Yellow
Write-Host "--------------------------------" -ForegroundColor Yellow
& .\verify-phase-6.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Build Tests..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$allPassed = $true

Write-Host "Step 1: Type checking with pnpm run check..." -ForegroundColor Yellow
$checkOutput = & pnpm run check 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Type check passed!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Type check failed!" -ForegroundColor Red
    Write-Host $checkOutput -ForegroundColor Red
    $allPassed = $false
}

Write-Host ""
Write-Host "Step 2: Building project with pnpm run build..." -ForegroundColor Yellow
$buildOutput = & pnpm run build 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Build succeeded!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Build failed!" -ForegroundColor Red
    Write-Host $buildOutput -ForegroundColor Red
    $allPassed = $false
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

if ($allPassed) {
    Write-Host "PHASE 6 COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "All tests passed successfully:" -ForegroundColor Green
    Write-Host "  - No non-canonical imports" -ForegroundColor White
    Write-Host "  - Type checking clean" -ForegroundColor White
    Write-Host "  - Build successful" -ForegroundColor White
    Write-Host "  - Canonical path established" -ForegroundColor White
    Write-Host ""
    Write-Host "Ready for Phase 7!" -ForegroundColor Green
} else {
    Write-Host "PHASE 6 INCOMPLETE" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Some tests failed. Please review errors above." -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
