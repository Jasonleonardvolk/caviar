# Simple Git Recovery Script
# D:\Dev\kha\tools\git\Simple-Recovery.ps1

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "D:\Dev\kha_backup_$ts"
$cleanDir = "D:\Dev\kha_clean_$ts"

Write-Host "=== SIMPLE GIT RECOVERY ===" -ForegroundColor Green
Write-Host "Timestamp: $ts" -ForegroundColor Cyan
Write-Host ""

# Step 1: Save important files (simple copy, no robocopy)
Write-Host "[1/5] Backing up your work files..." -ForegroundColor Yellow
Write-Host "  Backup location: $backupDir"

New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Copy important directories and files (skip the problematic ones)
$itemsToCopy = @(
    "tori_ui_svelte",
    "tools",
    "content",
    "*.py",
    "*.ps1",
    "*.md",
    "*.txt",
    "*.json",
    "*.toml",
    "*.yaml",
    "*.yml",
    ".gitignore",
    "requirements.txt"
)

foreach ($item in $itemsToCopy) {
    $source = Join-Path "D:\Dev\kha" $item
    if (Test-Path $source) {
        Write-Host "  Copying: $item" -ForegroundColor Gray
        
        # Check if it's a directory or file pattern
        if ($item.Contains("*")) {
            # It's a wildcard pattern
            Copy-Item $source -Destination $backupDir -Force -ErrorAction SilentlyContinue
        } else {
            # It's a specific file or directory
            $dest = Join-Path $backupDir $item
            if (Test-Path $source -PathType Container) {
                # It's a directory
                Copy-Item $source -Destination $dest -Recurse -Force -ErrorAction SilentlyContinue
            } else {
                # It's a file
                Copy-Item $source -Destination $dest -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Write-Host "  Backup complete!" -ForegroundColor Green

# Step 2: Clone fresh repository
Write-Host ""
Write-Host "[2/5] Cloning fresh repository..." -ForegroundColor Yellow
Write-Host "  Target: $cleanDir"

git clone https://github.com/Jasonleonardvolk/Tori.git $cleanDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Git clone failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Switch to feature branch
Write-Host ""
Write-Host "[3/5] Setting up feature branch..." -ForegroundColor Yellow
Push-Location $cleanDir

git checkout -b feat/wowpack-prores-hdr10-pipeline
if ($LASTEXITCODE -ne 0) {
    # Branch might already exist on remote, try checking it out
    git checkout feat/wowpack-prores-hdr10-pipeline
}

# Configure git settings
git config core.autocrlf false
git config core.eol lf

Write-Host "  Branch configured!" -ForegroundColor Green

# Step 4: Restore your work
Write-Host ""
Write-Host "[4/5] Restoring your work files..." -ForegroundColor Yellow

# Copy back the saved files
Get-ChildItem $backupDir | ForEach-Object {
    $destPath = Join-Path $cleanDir $_.Name
    Write-Host "  Restoring: $($_.Name)" -ForegroundColor Gray
    
    if ($_.PSIsContainer) {
        # Remove existing directory if it exists (except .git)
        if ($_.Name -ne ".git" -and (Test-Path $destPath)) {
            Remove-Item $destPath -Recurse -Force -ErrorAction SilentlyContinue
        }
        Copy-Item $_.FullName -Destination $destPath -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        Copy-Item $_.FullName -Destination $destPath -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "  Restoration complete!" -ForegroundColor Green

# Step 5: Check status
Write-Host ""
Write-Host "[5/5] Checking repository status..." -ForegroundColor Yellow

# Add all changes
git add .

# Show status
$status = git status --short
if ($status) {
    Write-Host "  Files ready to commit:" -ForegroundColor Green
    $status | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
} else {
    Write-Host "  No changes detected" -ForegroundColor Yellow
}

Pop-Location

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RECOVERY COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Clean repository: $cleanDir" -ForegroundColor Yellow
Write-Host "Backup location: $backupDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. cd $cleanDir" -ForegroundColor White
Write-Host "  2. git status" -ForegroundColor White
Write-Host "  3. git commit -m `"feat(wowpack): Recovery after corruption`"" -ForegroundColor White
Write-Host "  4. git push -u origin feat/wowpack-prores-hdr10-pipeline" -ForegroundColor White
Write-Host ""
Write-Host "After verifying everything works:" -ForegroundColor Cyan
Write-Host "  5. Rename D:\Dev\kha to D:\Dev\kha.OLD" -ForegroundColor White
Write-Host "  6. Rename $cleanDir to D:\Dev\kha" -ForegroundColor White