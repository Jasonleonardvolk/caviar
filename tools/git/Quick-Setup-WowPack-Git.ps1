# Quick-Setup-WowPack-Git.ps1
# Faster version that skips the time-consuming renormalize step
# Use this if you've already normalized line endings or want to do it separately

$ErrorActionPreference = "Continue"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     WOW Pack v1 Git Setup (Quick Mode)        " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check we're in the right place
if (-not (Test-Path ".git")) {
    Write-Host "[ERROR] Not in a Git repository!" -ForegroundColor Red
    Write-Host "Run this from D:\Dev\kha" -ForegroundColor Yellow
    exit 1
}

$currentBranch = git branch --show-current
Write-Host "Current branch: $currentBranch" -ForegroundColor Gray
Write-Host ""

# Step 1: Configure Git settings
Write-Host "[1/5] Configuring Git settings..." -ForegroundColor Yellow
git config core.autocrlf false
git config core.eol lf
Write-Host "  [OK] Set core.autocrlf=false, core.eol=lf" -ForegroundColor Green

# Step 2: Install Git hooks
Write-Host ""
Write-Host "[2/5] Installing Git hooks..." -ForegroundColor Yellow
$hooksScript = ".\tools\git\Setup-GitHooks.ps1"
if (Test-Path $hooksScript) {
    & $hooksScript
} else {
    Write-Host "  [WARNING] Setup-GitHooks.ps1 not found" -ForegroundColor Yellow
}

# Step 3: Stage the pipeline files
Write-Host ""
Write-Host "[3/5] Staging WOW Pack files..." -ForegroundColor Yellow

$filesToAdd = @(
    ".gitattributes",
    ".gitignore",
    "content/wowpack/ProRes-HDR-Pipeline.md",
    "content/wowpack/README.md",
    "tools/encode/Build-WowPack.ps1",
    "tools/encode/Batch-Encode-Simple.ps1",
    "tools/encode/Check-ProRes-Masters.ps1",
    "tools/release/Verify-WowPack.ps1",
    "tools/git/hooks/pre-commit.ps1",
    "tools/git/Setup-GitHooks.ps1",
    "tools/git/Setup-WowPack-Git.ps1",
    "tools/git/Setup-Git-LFS.ps1",
    ".github/workflows/wowpack.yml",
    "tori_ui_svelte/src/lib/video/chooseSource.ts",
    "tori_ui_svelte/src/lib/video/wowManifest.ts",
    "tori_ui_svelte/src/lib/show/ShowController.ts",
    "tori_ui_svelte/static/media/wow/wow.manifest.json",
    "tori_ui_svelte/src/lib/overlays/presets/EdgeGlow.json",
    "tori_ui_svelte/src/lib/overlays/presets/FresnelRim.json",
    "tori_ui_svelte/src/lib/overlays/presets/Bloom.json"
)

$staged = 0
foreach ($file in $filesToAdd) {
    if (Test-Path $file) {
        git add $file 2>$null
        if ($LASTEXITCODE -eq 0) {
            $staged++
            Write-Host "  [OK] $file" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "  Staged: $staged files" -ForegroundColor Green

# Step 4: Check for large files
Write-Host ""
Write-Host "[4/5] Checking for large files..." -ForegroundColor Yellow

# Use git ls-files with proper handling for quoted paths
$largeFiles = @()
$errorCount = 0

# Get staged files with size information
git ls-files --cached -z | ForEach-Object {
    if ($_ -and $_.Length -gt 0) {
        $path = $_
        
        # Handle quoted paths (Git quotes paths with special characters)
        if ($path.StartsWith('"') -and $path.EndsWith('"')) {
            # Remove quotes and unescape
            $path = $path.Substring(1, $path.Length - 2)
            $path = $path -replace '\\(.)', '$1'
        }
        
        # Skip if path contains illegal characters we can't handle
        try {
            if (Test-Path $path -ErrorAction SilentlyContinue) {
                $size = (Get-Item $path -ErrorAction SilentlyContinue).Length
                if ($size -gt 75MB) {
                    $largeFiles += [PSCustomObject]@{
                        Path = $path
                        SizeMB = [math]::Round($size/1MB, 2)
                    }
                }
            }
        } catch {
            # Silently skip files we can't process
            $errorCount++
        }
    }
}

if ($largeFiles.Count -gt 0) {
    Write-Host "  [WARNING] Large files detected and will be unstaged:" -ForegroundColor Red
    $largeFiles | ForEach-Object {
        Write-Host "    - $($_.Path) ($($_.SizeMB) MB)" -ForegroundColor Red
        # Try to unstage, but don't fail if it doesn't work
        git reset HEAD "$($_.Path)" 2>$null
    }
} else {
    Write-Host "  [OK] No large files staged" -ForegroundColor Green
    if ($errorCount -gt 0) {
        Write-Host "  [INFO] Skipped $errorCount files with special characters" -ForegroundColor Gray
    }
}

# Step 5: Show status
Write-Host ""
Write-Host "[5/5] Git status..." -ForegroundColor Yellow
Write-Host ""

# Get a cleaner status output
$modifiedFiles = git diff --cached --name-only 2>$null | Measure-Object -Line
$untrackedFiles = git ls-files --others --exclude-standard 2>$null | Measure-Object -Line

Write-Host "  Staged files: $($modifiedFiles.Lines)" -ForegroundColor Green
Write-Host "  Untracked files: $($untrackedFiles.Lines)" -ForegroundColor Gray

# Show abbreviated status
git status --short | Select-Object -First 20
$totalFiles = (git status --short | Measure-Object -Line).Lines
if ($totalFiles -gt 20) {
    Write-Host "  ... and $($totalFiles - 20) more files" -ForegroundColor Gray
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Quick Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "SKIPPED: Line ending normalization (do separately if needed)" -ForegroundColor Yellow
Write-Host "  To normalize later: git add --renormalize ." -ForegroundColor Gray
Write-Host ""
Write-Host "Ready to commit:" -ForegroundColor Green
Write-Host '  git commit -m "feat(wowpack): ProRes to HDR10/AV1/SDR pipeline"' -ForegroundColor White
Write-Host ""
Write-Host "Then push:" -ForegroundColor Green
Write-Host "  git push -u origin $currentBranch" -ForegroundColor White
