# Setup-GitHooks.ps1
# Installs Git hooks for the WOW Pack repository

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     Git Hooks Setup for WOW Pack              " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Find the repo root by looking for .git directory
$currentPath = Get-Location
$repoRoot = $currentPath

# Walk up the directory tree to find .git
while ($repoRoot -and !(Test-Path (Join-Path $repoRoot ".git"))) {
    $parent = Split-Path $repoRoot -Parent
    if ($parent -eq $repoRoot) {
        # Reached root of filesystem
        $repoRoot = $null
        break
    }
    $repoRoot = $parent
}

if (-not $repoRoot) {
    # Try current directory as fallback
    if (Test-Path ".git") {
        $repoRoot = Get-Location
    } else {
        Write-Host "[ERROR] Not in a Git repository!" -ForegroundColor Red
        Write-Host "Run this from within the repository." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Repository root: $repoRoot" -ForegroundColor Gray

$hooksSource = Join-Path $repoRoot "tools\git\hooks"
$gitHooksDir = Join-Path $repoRoot ".git\hooks"

# Create hooks directory if it doesn't exist
if (-not (Test-Path $gitHooksDir)) {
    New-Item -ItemType Directory -Path $gitHooksDir -Force | Out-Null
    Write-Host "Created .git\hooks directory" -ForegroundColor Green
}

# Install pre-commit hook
$preCommitSource = Join-Path $hooksSource "pre-commit.ps1"
$preCommitDest = Join-Path $gitHooksDir "pre-commit"

if (Test-Path $preCommitSource) {
    # Create a shell wrapper for PowerShell hook
    $wrapperContent = @"
#!/bin/sh
# Git hook wrapper to run PowerShell script
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "`$(git rev-parse --show-toplevel)/tools/git/hooks/pre-commit.ps1"
exit `$?
"@
    
    # Write the wrapper with Unix line endings
    $wrapperContent -replace "`r`n", "`n" | Set-Content -Path $preCommitDest -Encoding ASCII -NoNewline
    
    Write-Host "[OK] Installed pre-commit hook" -ForegroundColor Green
    Write-Host "     Blocks files > 75MB from being committed" -ForegroundColor Gray
    
    # Also create a Windows batch version for compatibility
    $batchWrapper = @"
@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\tools\git\hooks\pre-commit.ps1"
exit /b %ERRORLEVEL%
"@
    
    $batchPath = Join-Path $gitHooksDir "pre-commit.bat"
    $batchWrapper | Set-Content -Path $batchPath -Encoding ASCII
    
} else {
    Write-Host "[WARNING] pre-commit.ps1 not found at: $preCommitSource" -ForegroundColor Yellow
}

# Make hooks executable (on Unix-like systems)
if ($IsLinux -or $IsMacOS) {
    & chmod +x $preCommitDest 2>$null
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Hook Features:" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pre-Commit Hook:" -ForegroundColor Yellow
Write-Host "  - Prevents committing files > 75MB" -ForegroundColor Gray
Write-Host "  - Suggests using content\wowpack\input\ for media" -ForegroundColor Gray
Write-Host "  - Shows how to unstage large files" -ForegroundColor Gray
Write-Host ""
Write-Host "Test the hook:" -ForegroundColor Yellow
Write-Host "  1. Create a large test file:" -ForegroundColor Gray
Write-Host "     fsutil file createnew test-large.bin 80000000" -ForegroundColor White
Write-Host "  2. Try to commit it:" -ForegroundColor Gray
Write-Host "     git add test-large.bin && git commit -m 'test'" -ForegroundColor White
Write-Host "  3. Hook should block the commit" -ForegroundColor Gray
Write-Host "  4. Clean up:" -ForegroundColor Gray
Write-Host "     git reset HEAD test-large.bin && del test-large.bin" -ForegroundColor White
Write-Host ""
Write-Host "[OK] Git hooks setup complete!" -ForegroundColor Green
