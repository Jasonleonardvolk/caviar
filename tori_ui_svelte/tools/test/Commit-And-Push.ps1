# Commit-And-Push.ps1
# Commits all our configuration fixes and pushes with asset skip

param(
    [switch]$SkipAssets
)

Write-Host "=== COMMITTING CONFIGURATION FIXES ===" -ForegroundColor Cyan
Write-Host ""

Set-Location D:\Dev\kha\tori_ui_svelte

# Show current status
Write-Host "Current status:" -ForegroundColor Yellow
git status --short

Write-Host ""
Write-Host "Committing all changes..." -ForegroundColor Yellow

# Add all the configuration changes
git add package.json
git add svelte.config.js
git add svelte.config.js.bak
git add tools/release/Reset-And-Ship.ps1
git add tools/release/Verify-EndToEnd.ps1
git add tools/lint/
git add tools/runtime/
git add tools/test/
git add vite.config.mjs
git add vite.config.js.bak
git rm vite.config.js 2>$null

# Commit
$commitMsg = "fix: ESM configuration and add hardening scripts

- Fixed vite.config.js -> vite.config.mjs for ESM
- Removed BOM from package.json
- Added port guard to Reset-And-Ship.ps1
- Added BOM linter to Verify-EndToEnd.ps1
- Added enhanced Run-Mock.ps1 with better controls
- Added Check-BOM.ps1 lint tool
- Added various helper scripts for Git and process management"

git commit -m $commitMsg

Write-Host ""
Write-Host "Changes committed!" -ForegroundColor Green

# Push
Write-Host ""
if ($SkipAssets) {
    Write-Host "Pushing with IRIS_SKIP_ASSETS=1..." -ForegroundColor Yellow
    $env:IRIS_SKIP_ASSETS = "1"
} else {
    Write-Host "Pushing (with asset validation)..." -ForegroundColor Yellow
}

git push

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Push successful!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Push failed. Try with -SkipAssets flag:" -ForegroundColor Red
    Write-Host "  .\Commit-And-Push.ps1 -SkipAssets" -ForegroundColor Yellow
}
