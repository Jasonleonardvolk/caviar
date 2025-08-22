#!/usr/bin/env pwsh
# Test-PrePushHook.ps1
# Tests the pre-push hook without actually pushing

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "    Testing Pre-Push Hook" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$projectDir = "D:\Dev\kha\tori_ui_svelte"
Set-Location $projectDir

# Test 1: Run verifier directly
Write-Host "Test 1: Running verifier directly..." -ForegroundColor Yellow
& powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\release\verify-setup.ps1"
$verifierExit = $LASTEXITCODE
Write-Host "Verifier exit code: $verifierExit" -ForegroundColor Cyan

if ($verifierExit -eq 0) {
    Write-Host "[OK] Verifier passed" -ForegroundColor Green
} else {
    Write-Host "[X] Verifier failed with exit $verifierExit" -ForegroundColor Red
}
Write-Host ""

# Test 2: Run the hook script directly
Write-Host "Test 2: Running hook script..." -ForegroundColor Yellow
if (Test-Path ".githooks\pre-push") {
    # Use Git Bash to run the shell script
    $gitBash = "C:\Program Files\Git\bin\bash.exe"
    if (Test-Path $gitBash) {
        & $gitBash -c "./.githooks/pre-push"
        $hookExit = $LASTEXITCODE
        Write-Host "Hook exit code: $hookExit" -ForegroundColor Cyan
        
        if ($hookExit -eq 0) {
            Write-Host "[OK] Hook passed" -ForegroundColor Green
        } else {
            Write-Host "[X] Hook failed with exit $hookExit" -ForegroundColor Red
        }
    } else {
        Write-Host "[!] Git Bash not found at expected location" -ForegroundColor Yellow
        Write-Host "    Try: sh .githooks/pre-push" -ForegroundColor Gray
    }
} else {
    Write-Host "[X] Hook not found at .githooks\pre-push" -ForegroundColor Red
}
Write-Host ""

# Test 3: Test with skip assets flag
Write-Host "Test 3: Testing with IRIS_SKIP_ASSETS=1..." -ForegroundColor Yellow
$env:IRIS_SKIP_ASSETS = "1"
& powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\release\verify-setup.ps1" -SkipAssets
$skipExit = $LASTEXITCODE
$env:IRIS_SKIP_ASSETS = $null
Write-Host "Skip assets exit code: $skipExit" -ForegroundColor Cyan

if ($skipExit -eq 0) {
    Write-Host "[OK] Skip assets mode works" -ForegroundColor Green
} else {
    Write-Host "[X] Skip assets mode failed" -ForegroundColor Red
}
Write-Host ""

# Summary
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Yellow
if ($verifierExit -eq 0) {
    Write-Host "  Verifier: PASS" -ForegroundColor Green
} else {
    Write-Host "  Verifier: FAIL ($verifierExit)" -ForegroundColor Red
}

if ($hookExit -eq 0) {
    Write-Host "  Hook: PASS" -ForegroundColor Green
} elseif ($null -eq $hookExit) {
    Write-Host "  Hook: NOT TESTED" -ForegroundColor Yellow
} else {
    Write-Host "  Hook: FAIL ($hookExit)" -ForegroundColor Red
}

if ($skipExit -eq 0) {
    Write-Host "  Skip mode: PASS" -ForegroundColor Green
} else {
    Write-Host "  Skip mode: FAIL ($skipExit)" -ForegroundColor Red
}

Write-Host "=====================================" -ForegroundColor Cyan

# Overall result
if (($verifierExit -eq 0) -and (($hookExit -eq 0) -or ($null -eq $hookExit)) -and ($skipExit -eq 0)) {
    Write-Host "[OK] Pre-push hook is working correctly!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "[X] Pre-push hook has issues" -ForegroundColor Red
    Write-Host "Run: .\tools\git\Fix-GitHookLineEndings.ps1" -ForegroundColor Yellow
    exit 1
}