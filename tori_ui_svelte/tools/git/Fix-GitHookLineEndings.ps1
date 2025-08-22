#!/usr/bin/env pwsh
# Fix-GitHookLineEndings.ps1
# Ensures the pre-push hook has proper Unix line endings and is executable

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "   Fixing Git Hook Line Endings" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$projectDir = "D:\Dev\kha\tori_ui_svelte"
$hookPath = Join-Path $projectDir ".githooks\pre-push"

# Check if hook exists
if (-not (Test-Path $hookPath)) {
    Write-Host "[X] Hook not found at: $hookPath" -ForegroundColor Red
    exit 1
}

Write-Host "Processing: $hookPath" -ForegroundColor Yellow

# Read the file content
$content = Get-Content $hookPath -Raw

# Convert CRLF to LF
$contentLF = $content -replace "`r`n", "`n"

# Write back with Unix line endings
[System.IO.File]::WriteAllText($hookPath, $contentLF)

Write-Host "[OK] Converted to Unix line endings (LF)" -ForegroundColor Green

# Make executable in git
Set-Location $projectDir
git update-index --chmod=+x .githooks/pre-push 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Marked as executable in git" -ForegroundColor Green
} else {
    Write-Host "[!] Could not mark as executable (may already be set)" -ForegroundColor Yellow
}

# Stage the change if needed
$status = git status --porcelain .githooks/pre-push
if ($status) {
    Write-Host "[i] Hook has changes to commit" -ForegroundColor Yellow
    Write-Host "    Run: git add .githooks/pre-push" -ForegroundColor Gray
} else {
    Write-Host "[OK] Hook is ready" -ForegroundColor Green
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Done! Your hook should now work properly." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan