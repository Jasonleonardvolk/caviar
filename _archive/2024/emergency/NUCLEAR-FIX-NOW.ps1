#!/usr/bin/env pwsh
# NUCLEAR-FIX-NOW.ps1 - Complete solution to fix the Git mess

Write-Host "============================================" -ForegroundColor Red
Write-Host "    NUCLEAR FIX - COMPLETE SOLUTION" -ForegroundColor Yellow  
Write-Host "============================================" -ForegroundColor Red

# STEP 1: Reset the bad commit
Write-Host "`n[1/7] RESETTING bad commit..." -ForegroundColor Yellow
git reset --soft HEAD~1

# STEP 2: Unstage EVERYTHING
Write-Host "[2/7] UNSTAGING everything..." -ForegroundColor Yellow
git reset

# STEP 3: Remove ALL problematic files from Git tracking
Write-Host "[3/7] REMOVING all crap from Git tracking..." -ForegroundColor Red

# Google Drive temp files
git rm -r --cached .tmp.driveupload/ 2>$null
git rm -r --cached ".tmp.drivedownload*" 2>$null
git rm -r --cached "~*" 2>$null

# Node modules everywhere
git rm -r --cached node_modules/ 2>$null
git rm -r --cached tori_ui_svelte/node_modules/ 2>$null
git rm -r --cached frontend/hybrid/node_modules/ 2>$null

# Build outputs
git rm -r --cached tori_ui_svelte/.svelte-kit/ 2>$null
git rm -r --cached tori_ui_svelte/build/ 2>$null
git rm -r --cached tori_ui_svelte/dist/ 2>$null
git rm -r --cached tori_ui_svelte/.vite/ 2>$null
git rm -r --cached .turbo/ 2>$null
git rm -r --cached .tauri/ 2>$null

# Python stuff
git rm -r --cached __pycache__/ 2>$null
git rm -r --cached .venv/ 2>$null
git rm -r --cached venv/ 2>$null
git rm -r --cached "*.pyc" 2>$null

# Media files
git rm -r --cached "*.mp4" 2>$null
git rm -r --cached "*.mov" 2>$null
git rm -r --cached "*.m4s" 2>$null
git rm -r --cached content/wowpack/input/ 2>$null
git rm -r --cached content/wowpack/video/ 2>$null
git rm -r --cached content/wowpack/stills/ 2>$null
git rm -r --cached tori_ui_svelte/static/media/hls/ 2>$null
git rm -r --cached tori_ui_svelte/static/media/wow/*.mp4 2>$null

# Large binaries
git rm -r --cached "*.exe" 2>$null
git rm -r --cached "*.dll" 2>$null
git rm -r --cached "*.zip" 2>$null
git rm -r --cached "*.tar" 2>$null
git rm -r --cached "*.gz" 2>$null

# OS junk
git rm -r --cached "Thumbs.db" 2>$null
git rm -r --cached ".DS_Store" 2>$null
git rm -r --cached "Desktop.ini" 2>$null

# STEP 4: Update .gitignore to be bulletproof
Write-Host "[4/7] UPDATING .gitignore to be bulletproof..." -ForegroundColor Yellow

$gitignoreAdditions = @"

# === NUCLEAR EXCLUSIONS ===
# Google Drive sync files
.tmp.driveupload/
.tmp.drivedownload*/
~*
*.tmp

# All node_modules everywhere
**/node_modules/
node_modules/

# All Python environments
**/__pycache__/
**/*.pyc
**/venv/
**/.venv/

# All build outputs
**/.svelte-kit/
**/build/
**/dist/
**/.vite/
**/.turbo/
**/.tauri/

# All media files
*.mp4
*.mov
*.m4s
*.avi
*.mkv
*.webm

# All large binaries
*.exe
*.dll
*.zip
*.tar
*.gz
*.7z
*.rar

# WOW Pack build folders
content/wowpack/input/
content/wowpack/video/
content/wowpack/stills/
tori_ui_svelte/static/media/hls/
tori_ui_svelte/static/media/wow/*.mp4

# Keep only the manifest
!tori_ui_svelte/static/media/wow/wow.manifest.json
"@

# Append if not already there
$currentGitignore = Get-Content .gitignore -Raw
if ($currentGitignore -notmatch "NUCLEAR EXCLUSIONS") {
    Add-Content .gitignore $gitignoreAdditions
}

# STEP 5: Stage ONLY the essential WOW Pack files
Write-Host "[5/7] STAGING only essential WOW Pack code files..." -ForegroundColor Green

# Core Git files
git add .gitattributes
git add .gitignore

# WOW Pack documentation
$docs = @(
    "content/wowpack/ProRes-HDR-Pipeline.md",
    "content/wowpack/README.md"
)
foreach ($doc in $docs) {
    if (Test-Path $doc) { git add $doc }
}

# PowerShell tools
$psTools = Get-ChildItem -Path @(
    "tools/encode/*.ps1",
    "tools/release/*.ps1", 
    "tools/git/*.ps1"
) -ErrorAction SilentlyContinue
foreach ($tool in $psTools) {
    git add $tool.FullName
}

# TypeScript/JavaScript source files
$srcFiles = Get-ChildItem -Path @(
    "tori_ui_svelte/src/lib/video/*.ts",
    "tori_ui_svelte/src/lib/show/*.ts",
    "tori_ui_svelte/src/lib/overlays/*.ts",
    "tori_ui_svelte/src/lib/overlays/presets/*.json"
) -ErrorAction SilentlyContinue
foreach ($src in $srcFiles) {
    git add $src.FullName
}

# Manifest only
if (Test-Path "tori_ui_svelte/static/media/wow/wow.manifest.json") {
    git add "tori_ui_svelte/static/media/wow/wow.manifest.json"
}

# GitHub workflows
$workflows = Get-ChildItem ".github/workflows/*.yml" -ErrorAction SilentlyContinue
foreach ($wf in $workflows) {
    git add $wf.FullName
}

# STEP 6: Verify what we're committing
Write-Host "`n[6/7] VERIFYING commit size..." -ForegroundColor Cyan

$files = git diff --cached --name-only
$totalSize = 0
$fileCount = 0

foreach ($file in $files) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length
        $totalSize += $size
        $fileCount++
        
        # Alert if any single file is over 1MB
        if ($size -gt 1MB) {
            $sizeMB = [math]::Round($size / 1MB, 2)
            Write-Host "  WARNING: $file is $sizeMB MB" -ForegroundColor Yellow
        }
    }
}

$totalSizeMB = [math]::Round($totalSize / 1MB, 2)
Write-Host "`n  Files to commit: $fileCount" -ForegroundColor Green
Write-Host "  Total size: $totalSizeMB MB" -ForegroundColor Green

if ($totalSizeMB -gt 50) {
    Write-Host "`n  ERROR: Still too large! Something's wrong." -ForegroundColor Red
    Write-Host "  Run 'git status' to see what's staged" -ForegroundColor Yellow
    exit 1
}

# STEP 7: Show final status
Write-Host "`n[7/7] FINAL STATUS:" -ForegroundColor Green
git status --short

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "         READY TO COMMIT!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "`nTotal size: $totalSizeMB MB (should be < 10 MB)" -ForegroundColor Cyan
Write-Host "`nNEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Run: git commit -m `"feat(wowpack): ProRes to HDR10/AV1/SDR pipeline`"" -ForegroundColor White
Write-Host "  2. Run: git push -u origin feat/test-workflow-demo" -ForegroundColor White
Write-Host "`nIf push still fails, run:" -ForegroundColor Yellow
Write-Host "  git status" -ForegroundColor White
Write-Host "  git diff --cached --stat" -ForegroundColor White
