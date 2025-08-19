# Setup-WowPack-Git.ps1
# Complete Git setup for WOW Pack v1 release
# Run from D:\Dev\kha

$ErrorActionPreference = "Continue"  # Continue on non-fatal errors

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     WOW Pack v1 Git Setup                     " -ForegroundColor Cyan
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
Write-Host "[1/7] Configuring Git settings..." -ForegroundColor Yellow
git config core.autocrlf false
git config core.eol lf
Write-Host "  [OK] Set core.autocrlf=false, core.eol=lf" -ForegroundColor Green

# Step 2: Install Git hooks
Write-Host ""
Write-Host "[2/7] Installing Git hooks..." -ForegroundColor Yellow
$hooksScript = ".\tools\git\Setup-GitHooks.ps1"
if (Test-Path $hooksScript) {
    & $hooksScript
} else {
    Write-Host "  [WARNING] Setup-GitHooks.ps1 not found" -ForegroundColor Yellow
}

# Step 3: Renormalize line endings
Write-Host ""
Write-Host "[3/7] Renormalizing line endings..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes for large repositories..." -ForegroundColor Gray
Write-Host "  Processing files..." -ForegroundColor Gray

# Show a simple progress indicator
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$job = Start-Job -ScriptBlock { 
    Set-Location $using:PWD
    git add --renormalize . 2>&1
}

while ($job.State -eq 'Running') {
    Write-Host "." -NoNewline -ForegroundColor Gray
    Start-Sleep -Milliseconds 500
    if ($stopwatch.Elapsed.TotalSeconds -gt 60) {
        Write-Host ""
        Write-Host "  Still processing (this is normal for large repos)..." -ForegroundColor Yellow
        $stopwatch.Restart()
    }
}

$result = Receive-Job -Job $job
Remove-Job -Job $job
$stopwatch.Stop()

Write-Host ""  # New line after dots

if ($job.State -eq 'Completed') {
    $changedFiles = git diff --cached --name-only 2>$null | Measure-Object -Line
    if ($changedFiles.Lines -gt 0) {
        Write-Host "  [OK] Line endings normalized for $($changedFiles.Lines) files" -ForegroundColor Green
    } else {
        Write-Host "  [OK] Line endings checked (no changes needed)" -ForegroundColor Green
    }
    Write-Host "  Time taken: $([math]::Round($stopwatch.Elapsed.TotalSeconds, 1)) seconds" -ForegroundColor Gray
} else {
    Write-Host "  [INFO] Line ending normalization completed" -ForegroundColor Gray
}

# Step 4: Create feature branch
Write-Host ""
Write-Host "[4/7] Creating feature branch..." -ForegroundColor Yellow
$branchName = "feat/wowpack-prores-hdr10-pipeline"

# Check if branch already exists
$existingBranch = git branch --list $branchName 2>$null
if ($existingBranch) {
    Write-Host "  Branch '$branchName' already exists" -ForegroundColor Yellow
    $answer = Read-Host "  Switch to it? (y/n)"
    if ($answer -eq 'y') {
        git checkout $branchName
        Write-Host "  [OK] Switched to: $branchName" -ForegroundColor Green
    } else {
        Write-Host "  [INFO] Staying on current branch: $currentBranch" -ForegroundColor Gray
    }
} else {
    git checkout -b $branchName 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Created and switched to: $branchName" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Could not create branch (may already exist remotely)" -ForegroundColor Yellow
    }
}

# Step 5: Stage the pipeline files
Write-Host ""
Write-Host "[5/7] Staging WOW Pack files..." -ForegroundColor Yellow

$filesToAdd = @(
    ".gitattributes",
    ".gitignore",
    "content/wowpack/ProRes-HDR-Pipeline.md",
    "content/wowpack/README.md",
    "tools/encode/Build-WowPack.ps1",
    "tools/encode/Batch-Encode-Simple.ps1",
    "tools/encode/Batch-Encode-All.ps1",
    "tools/encode/Check-ProRes-Masters.ps1",
    "tools/encode/Check-WowPack-Status.ps1",
    "tools/encode/Encode-All-Individual.ps1",
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
    "tori_ui_svelte/src/lib/overlays/presets/Bloom.json",
    "tori_ui_svelte/src/lib/show/recipes/wow.nebula.json"
)

$staged = 0
$missing = @()

foreach ($file in $filesToAdd) {
    if (Test-Path $file) {
        git add $file 2>$null
        if ($LASTEXITCODE -eq 0) {
            $staged++
            Write-Host "  [OK] $file" -ForegroundColor Green
        } else {
            Write-Host "  [WARN] Could not stage: $file" -ForegroundColor Yellow
        }
    } else {
        $missing += $file
        Write-Host "  [SKIP] $file (not found)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "  Staged: $staged files" -ForegroundColor Green
if ($missing.Count -gt 0) {
    Write-Host "  Missing: $($missing.Count) files (this is OK)" -ForegroundColor Gray
}

# Step 6: Check for large files
Write-Host ""
Write-Host "[6/7] Checking for accidentally staged large files..." -ForegroundColor Yellow

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
    Write-Host "  [WARNING] Large files detected:" -ForegroundColor Red
    $largeFiles | ForEach-Object {
        Write-Host "    - $($_.Path) ($($_.SizeMB) MB)" -ForegroundColor Red
        git reset HEAD "$($_.Path)" 2>$null
    }
    Write-Host "  [OK] Large files unstaged" -ForegroundColor Green
} else {
    Write-Host "  [OK] No large files staged" -ForegroundColor Green
    if ($errorCount -gt 0) {
        Write-Host "  [INFO] Skipped $errorCount files with special characters" -ForegroundColor Gray
    }
}

# Step 7: Show status
Write-Host ""
Write-Host "[7/7] Git status..." -ForegroundColor Yellow
Write-Host ""
git status --short

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Review staged files:" -ForegroundColor White
Write-Host "   git status" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Commit the changes:" -ForegroundColor White
Write-Host '   git commit -m "feat(wowpack): ProRes to HDR10/AV1/SDR pipeline + manifest + runtime source selection"' -ForegroundColor Gray
Write-Host ""
Write-Host "3. Push the branch:" -ForegroundColor White
$currentRemoteBranch = git branch --show-current
Write-Host "   git push -u origin $currentRemoteBranch" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Open a PR on GitHub/GitLab" -ForegroundColor White
Write-Host ""
Write-Host "Optional - Add more specific commits:" -ForegroundColor Yellow
Write-Host '   git commit -m "docs(wowpack): add ProRes-HDR-Pipeline.md with color pipeline"' -ForegroundColor Gray
Write-Host '   git commit -m "chore(git): enforce EOL, ignore large media, add pre-commit guard"' -ForegroundColor Gray
Write-Host '   git commit -m "ci(github): add wowpack verification workflow"' -ForegroundColor Gray
Write-Host ""
Write-Host "For release (after merge to main):" -ForegroundColor Yellow
Write-Host '   git checkout main' -ForegroundColor Gray
Write-Host '   git pull' -ForegroundColor Gray
Write-Host '   git tag -a wowpack-v1.0.0 -m "WOW Pack v1: ProRes to HDR10/AV1/SDR pipeline"' -ForegroundColor Gray
Write-Host '   git push origin wowpack-v1.0.0' -ForegroundColor Gray
