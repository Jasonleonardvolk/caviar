# Setup-Git-LFS.ps1
# Optional: Enable Git LFS for versioning final show-floor clips
# Only run this if you need to version large media files

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     Git LFS Setup for Media Files             " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "WARNING: Git LFS has bandwidth quotas!" -ForegroundColor Yellow
Write-Host "  - GitHub: 1GB storage, 1GB/month bandwidth (free)" -ForegroundColor Gray
Write-Host "  - Consider costs before enabling for large files" -ForegroundColor Gray
Write-Host ""

$answer = Read-Host "Enable Git LFS for .mp4 and .mov files? (y/n)"
if ($answer -ne 'y') {
    Write-Host "LFS setup cancelled." -ForegroundColor Yellow
    exit
}

# Check if Git LFS is installed
$lfsInstalled = Get-Command git-lfs -ErrorAction SilentlyContinue
if (-not $lfsInstalled) {
    Write-Host "[ERROR] Git LFS not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Git LFS first:" -ForegroundColor Yellow
    Write-Host "  winget install GitHub.GitLFS" -ForegroundColor White
    Write-Host "  OR download from: https://git-lfs.github.com/" -ForegroundColor Gray
    exit 1
}

Write-Host "[1/4] Initializing Git LFS..." -ForegroundColor Yellow
git lfs install
Write-Host "  [OK] Git LFS initialized" -ForegroundColor Green

Write-Host ""
Write-Host "[2/4] Tracking media files..." -ForegroundColor Yellow
git lfs track "*.mp4"
git lfs track "*.mov"
git lfs track "*.m4s"
Write-Host "  [OK] Tracking .mp4, .mov, .m4s files" -ForegroundColor Green

Write-Host ""
Write-Host "[3/4] Updating .gitattributes..." -ForegroundColor Yellow
git add .gitattributes
Write-Host "  [OK] .gitattributes updated with LFS rules" -ForegroundColor Green

Write-Host ""
Write-Host "[4/4] Checking LFS status..." -ForegroundColor Yellow
git lfs ls-files

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Git LFS Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "LFS is now tracking:" -ForegroundColor Yellow
Write-Host "  - *.mp4 files" -ForegroundColor Gray
Write-Host "  - *.mov files" -ForegroundColor Gray
Write-Host "  - *.m4s files" -ForegroundColor Gray
Write-Host ""
Write-Host "To add a final clip to version control:" -ForegroundColor Yellow
Write-Host "  1. Encode it first:" -ForegroundColor Gray
Write-Host '     .\tools\encode\Build-WowPack.ps1 -Basename "final_hero" -Input "path\to\master.mov"' -ForegroundColor White
Write-Host "  2. Add ONLY the final runtime version:" -ForegroundColor Gray
Write-Host '     git add tori_ui_svelte\static\media\wow\final_hero_hdr10.mp4' -ForegroundColor White
Write-Host '  3. Commit with LFS:' -ForegroundColor Gray
Write-Host '     git commit -m "feat(media): add final hero clip for Dallas show"' -ForegroundColor White
Write-Host ""
Write-Host "Important:" -ForegroundColor Red
Write-Host "  - Do NOT add content\wowpack\input\*.mov (source masters)" -ForegroundColor Gray
Write-Host "  - Do NOT add content\wowpack\video\*.mp4 (archives)" -ForegroundColor Gray
Write-Host "  - ONLY add final runtime assets when necessary" -ForegroundColor Gray
Write-Host ""
Write-Host "Check LFS quota usage:" -ForegroundColor Yellow
Write-Host "  git lfs ls-files --size" -ForegroundColor White
