#!/usr/bin/env pwsh
# Convert-HookToUnix.ps1
# Ensures proper Unix line endings and removes Windows-specific issues

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "   Converting Hook to Unix Format" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$projectDir = "D:\Dev\kha\tori_ui_svelte"
$hookPath = Join-Path $projectDir ".githooks\pre-push"

if (-not (Test-Path $hookPath)) {
    Write-Host "[X] Hook not found at: $hookPath" -ForegroundColor Red
    exit 1
}

# Read the current hook content
$content = Get-Content $hookPath -Raw

# Remove any Windows-specific characters and fix line endings
$fixedContent = $content -replace "`r`n", "`n"  # CRLF to LF
$fixedContent = $fixedContent -replace "`r", "`n"  # Any standalone CR to LF

# Write with UTF-8 no BOM and Unix line endings
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($hookPath, $fixedContent, $utf8NoBom)

Write-Host "[OK] Converted to Unix format (LF)" -ForegroundColor Green

# Make executable in git
Set-Location $projectDir
git update-index --chmod=+x .githooks/pre-push 2>$null

Write-Host "[OK] Marked as executable" -ForegroundColor Green

# Stage the changes
git add .githooks/pre-push
Write-Host "[OK] Staged changes" -ForegroundColor Green

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Hook is now Unix-compatible!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan