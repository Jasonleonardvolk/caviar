# Setup-GitHooks.ps1
# Configure Git to use versioned hooks in .githooks directory

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "    iRis Git Hooks Configuration" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Find repository root
$repoRoot = git rev-parse --show-toplevel 2>$null
if (!$repoRoot) {
    Write-Host "ERROR: Not in a Git repository" -ForegroundColor Red
    exit 1
}

# Convert to Windows path
$repoRoot = $repoRoot -replace '/', '\'
Set-Location $repoRoot

Write-Host "Repository: $repoRoot" -ForegroundColor Green
Write-Host ""

# Check current hooks path
$currentHooksPath = git config core.hooksPath
if ($currentHooksPath) {
    Write-Host "Current hooks path: $currentHooksPath" -ForegroundColor Yellow
} else {
    Write-Host "No custom hooks path configured (using .git/hooks)" -ForegroundColor Gray
}

# Configure to use .githooks
Write-Host "`nConfiguring Git to use .githooks directory..." -ForegroundColor Cyan
git config core.hooksPath .githooks

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Git hooks path set to .githooks" -ForegroundColor Green
} else {
    Write-Host "[X] Failed to set hooks path" -ForegroundColor Red
    exit 1
}

# Verify hooks exist
$hooksDir = Join-Path $repoRoot ".githooks"
if (!(Test-Path $hooksDir)) {
    Write-Host "Creating .githooks directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $hooksDir -Force | Out-Null
}

# Check for pre-push hook
$prePushHook = Join-Path $hooksDir "pre-push"
$prePushCmd = Join-Path $hooksDir "pre-push.cmd"

if (Test-Path $prePushHook) {
    Write-Host "[OK] pre-push hook exists (Unix/Git Bash)" -ForegroundColor Green
} else {
    Write-Host "[!] pre-push hook not found" -ForegroundColor Yellow
}

if (Test-Path $prePushCmd) {
    Write-Host "[OK] pre-push.cmd exists (Windows CMD)" -ForegroundColor Green
} else {
    Write-Host "[!] pre-push.cmd not found" -ForegroundColor Yellow
}

# Make hooks executable (for Git Bash/WSL)
if (Test-Path $prePushHook) {
    # Set executable bit in Git index
    git update-index --chmod=+x .githooks/pre-push 2>$null
    Write-Host "[OK] pre-push hook marked as executable" -ForegroundColor Green
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "        Configuration Complete" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The following will now happen automatically:" -ForegroundColor Yellow
Write-Host "  * Every 'git push' runs verification checks" -ForegroundColor White
Write-Host "  * Push is blocked if checks fail" -ForegroundColor White
Write-Host "  * Endpoints are tested if server is running" -ForegroundColor White
Write-Host ""
Write-Host "To test the hook manually:" -ForegroundColor Yellow
Write-Host "  .\.githooks\pre-push" -ForegroundColor White
Write-Host ""
Write-Host "To bypass hooks (emergency only):" -ForegroundColor Yellow
Write-Host "  git push --no-verify" -ForegroundColor White
Write-Host ""
Write-Host "To disable hooks:" -ForegroundColor Yellow
Write-Host "  git config --unset core.hooksPath" -ForegroundColor White
Write-Host ""