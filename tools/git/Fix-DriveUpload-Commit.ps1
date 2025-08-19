#!/usr/bin/env pwsh
# Fix-DriveUpload-Commit.ps1 - Emergency fix for Google Drive temp files in commit

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "EMERGENCY FIX: Remove Drive Upload Files" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Cyan

# Step 1: Reset the bad commit but keep changes
Write-Host "`n[1/5] Resetting bad commit (keeping your files)..." -ForegroundColor Yellow
git reset --soft HEAD~1

# Step 2: Unstage everything
Write-Host "[2/5] Unstaging all files..." -ForegroundColor Yellow
git reset

# Step 3: Remove the Google Drive temp files from Git tracking
Write-Host "[3/5] Removing .tmp.driveupload from Git tracking..." -ForegroundColor Yellow
git rm -r --cached .tmp.driveupload/ 2>$null

# Step 4: Stage ONLY the WOW Pack code files (no binaries!)
Write-Host "[4/5] Staging ONLY code files..." -ForegroundColor Green

# Git configuration files
git add .gitattributes .gitignore

# WOW Pack documentation
git add content/wowpack/*.md 2>$null

# Encoding tools
git add tools/encode/*.ps1 2>$null

# Release tools
git add tools/release/*.ps1 2>$null

# Git setup tools
git add tools/git/*.ps1 2>$null

# GitHub workflows if they exist
git add .github/workflows/*.yml 2>$null

# TypeScript/Svelte source files (code only)
git add tori_ui_svelte/src/lib/video/*.ts 2>$null
git add tori_ui_svelte/src/lib/show/*.ts 2>$null
git add tori_ui_svelte/src/lib/overlays/presets/*.json 2>$null

# Manifest file only (no actual videos)
git add tori_ui_svelte/static/media/wow/wow.manifest.json 2>$null

# Step 5: Show what we're about to commit
Write-Host "`n[5/5] Files staged for commit:" -ForegroundColor Cyan
git status --short

# Check total size
$totalSize = 0
git diff --cached --name-only | ForEach-Object {
    if (Test-Path $_) {
        $size = (Get-Item $_).Length
        $totalSize += $size
    }
}
$totalSizeMB = [math]::Round($totalSize / 1MB, 2)

Write-Host "`n===========================================" -ForegroundColor Green
Write-Host "Total size to commit: $totalSizeMB