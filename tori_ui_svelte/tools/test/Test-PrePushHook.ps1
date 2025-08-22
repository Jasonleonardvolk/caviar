# Test-PrePushHook.ps1
# Demonstrates the pre-push BOM detection gate

param(
    [switch]$Cleanup
)

Write-Host "=== TESTING PRE-PUSH BOM DETECTION ===" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the right directory
Set-Location D:\Dev\kha\tori_ui_svelte

if ($Cleanup) {
    Write-Host "Cleaning up test files..." -ForegroundColor Yellow
    Remove-Item bad.json -Force -ErrorAction SilentlyContinue
    git reset --hard HEAD 2>$null
    Write-Host "Cleanup complete" -ForegroundColor Green
    exit
}

# Step 1: Create a file with BOM
Write-Host "Step 1: Creating bad.json with BOM..." -ForegroundColor Yellow
[byte[]]$bom = 0xEF,0xBB,0xBF
$jsonContent = '{"test":true,"message":"This file has a BOM"}'
[System.IO.File]::WriteAllBytes("bad.json", $bom + [System.Text.Encoding]::UTF8.GetBytes($jsonContent))
Write-Host "  Created bad.json with BOM" -ForegroundColor Green

# Step 2: Try to commit and push (should fail)
Write-Host ""
Write-Host "Step 2: Attempting to commit and push (should FAIL)..." -ForegroundColor Yellow
git add bad.json
git commit -m "test: inject BOM (should fail pre-push)" 2>&1 | Out-Null

Write-Host "  Committed locally. Now pushing..." -ForegroundColor Cyan
$pushResult = git push 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  GOOD! Push was blocked by pre-push hook" -ForegroundColor Green
    Write-Host "  Error was: $pushResult" -ForegroundColor Gray
} else {
    Write-Host "  WARNING: Push succeeded (hook may not be configured)" -ForegroundColor Red
}

# Step 3: Fix the BOM
Write-Host ""
Write-Host "Step 3: Fixing the BOM..." -ForegroundColor Yellow
& powershell .\tools\lint\Check-BOM.ps1 -AutoFix
Write-Host "  BOM removed from bad.json" -ForegroundColor Green

# Step 4: Commit the fix and push (should succeed)
Write-Host ""
Write-Host "Step 4: Committing fix and pushing (should SUCCEED)..." -ForegroundColor Yellow
git add bad.json
git commit -m "chore: remove BOM from bad.json" 2>&1 | Out-Null

$pushResult2 = git push 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  SUCCESS! Push completed after BOM fix" -ForegroundColor Green
} else {
    Write-Host "  Push failed: $pushResult2" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== TEST COMPLETE ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To cleanup: .\Test-PrePushHook.ps1 -Cleanup" -ForegroundColor Gray
