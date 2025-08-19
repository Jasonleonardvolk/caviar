#!/usr/bin/env pwsh
# REAL-NUCLEAR-FIX.ps1 - ACTUALLY fixes the mess

Write-Host "============================================" -ForegroundColor Red
Write-Host "    REAL NUCLEAR FIX - CLEAN SLATE" -ForegroundColor Yellow  
Write-Host "============================================" -ForegroundColor Red

# STEP 1: ABORT! ABORT! Reset everything
Write-Host "`n[1/5] HARD RESET - Undoing the mess..." -ForegroundColor Red
git reset HEAD~1  # Go back before the bad commit
git reset         # Unstage EVERYTHING

# STEP 2: Clean status check
Write-Host "`n[2/5] Current status (should show lots of untracked files)..." -ForegroundColor Yellow
$stagedCount = (git diff --cached --name-only | Measure-Object).Count
Write-Host "  Staged files: $stagedCount (should be 0)" -ForegroundColor Cyan

# STEP 3: Add ONLY the WOW Pack files we actually created
Write-Host "`n[3/5] Adding ONLY the WOW Pack files..." -ForegroundColor Green

# Add the exact files for WOW Pack - NO WILDCARDS!
$filesToAdd = @(
    # Git configuration
    ".gitattributes",
    ".gitignore",
    
    # WOW Pack documentation  
    "content/wowpack/ProRes-HDR-Pipeline.md",
    
    # PowerShell tools we created
    "tools/encode/Build-WowPack.ps1",
    "tools/encode/Batch-Encode-Simple.ps1", 
    "tools/encode/Check-ProRes-Masters.ps1",
    "tools/release/Verify-WowPack.ps1",
    "tools/git/Setup-WowPack-Git.ps1",
    "tools/git/Quick-Setup-WowPack-Git.ps1",
    "tools/git/Ultra-Quick-Setup-WowPack-Git.ps1",
    "tools/git/Fix-DriveUpload-Commit.ps1",
    "NUCLEAR-FIX-NOW.ps1",
    "REAL-NUCLEAR-FIX.ps1",
    
    # The manifest file ONLY
    "tori_ui_svelte/static/media/wow/wow.manifest.json"
)

$addedFiles = @()
$missingFiles = @()

foreach ($file in $filesToAdd) {
    if (Test-Path $file) {
        git add $file 2>$null
        $addedFiles += $file
    } else {
        $missingFiles += $file
    }
}

Write-Host "`n  Added $($addedFiles.Count) files:" -ForegroundColor Green
foreach ($file in $addedFiles) {
    Write-Host "    + $file" -ForegroundColor Green
}

if ($missingFiles.Count -gt 0) {
    Write-Host "`n  Missing files (skipped):" -ForegroundColor Gray
    foreach ($file in $missingFiles) {
        Write-Host "    - $file" -ForegroundColor Gray
    }
}

# STEP 4: Verify size
Write-Host "`n[4/5] Verifying size..." -ForegroundColor Cyan
$totalSize = 0
$fileCount = 0

git diff --cached --name-only | ForEach-Object {
    if (Test-Path $_) {
        $size = (Get-Item $_).Length
        $totalSize += $size
        $fileCount++
    }
}

$totalSizeKB = [math]::Round($totalSize / 1KB, 2)
Write-Host "  Files staged: $fileCount" -ForegroundColor Green
Write-Host "  Total size: $totalSizeKB KB" -ForegroundColor Green

if ($totalSize -gt 10MB) {
    Write-Host "`n  WARNING: Still large! Check what got added:" -ForegroundColor Red
    git diff --cached --stat
    exit 1
}

# STEP 5: Final status
Write-Host "`n[5/5] Final staged files:" -ForegroundColor Green
git diff --cached --name-only | ForEach-Object {
    $size = 0
    if (Test-Path $_) {
        $size = (Get-Item $_).Length
        $sizeKB = [math]::Round($size / 1KB, 2)
        Write-Host "  $_ ($sizeKB KB)" -ForegroundColor Cyan
    }
}

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "         CLEAN AND READY!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "`nTotal: $totalSizeKB KB (perfect!)" -ForegroundColor Green
Write-Host "`nNOW RUN THESE COMMANDS:" -ForegroundColor Yellow
Write-Host "" -ForegroundColor White
Write-Host "  git commit -m `"feat(wowpack): ProRes to HDR10/AV1/SDR pipeline`"" -ForegroundColor White
Write-Host "  git push -u origin feat/test-workflow-demo" -ForegroundColor White
Write-Host "" -ForegroundColor White
