# Ultra-Quick-Setup-WowPack-Git.ps1
# Ultra-fast version - skips all slow checks
# Use this when you need to commit immediately

$ErrorActionPreference = "Continue"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     WOW Pack v1 Git Setup (ULTRA FAST)        " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check we're in the right place
if (-not (Test-Path ".git")) {
    Write-Host "[ERROR] Not in a Git repository!" -ForegroundColor Red
    exit 1
}

$currentBranch = git branch --show-current
Write-Host "Current branch: $currentBranch" -ForegroundColor Gray
Write-Host ""

# Step 1: Configure Git
Write-Host "[1/3] Configuring Git..." -ForegroundColor Yellow
git config core.autocrlf false
git config core.eol lf
Write-Host "  [OK] Git configured" -ForegroundColor Green

# Step 2: Stage WOW Pack files
Write-Host ""
Write-Host "[2/3] Staging WOW Pack files..." -ForegroundColor Yellow

# Just add the essential files directly
git add .gitattributes .gitignore 2>$null
git add content/wowpack/*.md 2>$null
git add tools/encode/*.ps1 2>$null
git add tools/release/*.ps1 2>$null
git add tools/git/*.ps1 2>$null
git add tools/git/hooks/*.ps1 2>$null
git add .github/workflows/*.yml 2>$null
git add tori_ui_svelte/src/lib/video/*.ts 2>$null
git add tori_ui_svelte/src/lib/show/*.ts 2>$null
git add tori_ui_svelte/src/lib/show/recipes/*.json 2>$null
git add tori_ui_svelte/src/lib/overlays/presets/*.json 2>$null
git add tori_ui_svelte/static/media/wow/wow.manifest.json 2>$null

Write-Host "  [OK] Files staged" -ForegroundColor Green

# Step 3: Quick status
Write-Host ""
Write-Host "[3/3] Status..." -ForegroundColor Yellow

$stagedCount = (git diff --cached --name-only 2>$null | Measure-Object -Line).Lines
Write-Host "  Staged files: $stagedCount" -ForegroundColor Green

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Ultra-Quick Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "SKIPPED:" -ForegroundColor Yellow
Write-Host "  - Line ending normalization" -ForegroundColor Gray
Write-Host "  - Large file checking" -ForegroundColor Gray
Write-Host "  - Git hooks installation" -ForegroundColor Gray
Write-Host ""
Write-Host "Ready to commit NOW:" -ForegroundColor Green
Write-Host '  git commit -m "feat(wowpack): ProRes to HDR10/AV1/SDR pipeline"' -ForegroundColor White
Write-Host ""
Write-Host "Push:" -ForegroundColor Green
Write-Host "  git push -u origin $currentBranch" -ForegroundColor White
Write-Host ""
Write-Host "Note: Run pre-commit check manually if needed:" -ForegroundColor Gray
Write-Host "  powershell .\tools\git\hooks\pre-commit.ps1" -ForegroundColor Gray
