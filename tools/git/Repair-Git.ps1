# Git Repository Repair Script
# Fixes corruption, missing objects, and broken links

param(
    [switch]$Aggressive,
    [switch]$BackupFirst
)

Write-Host "GIT REPOSITORY REPAIR TOOL" -ForegroundColor Cyan -BackgroundColor DarkBlue
Write-Host ("=" * 50)
Write-Host ""

$repoPath = "D:\Dev\kha"
Set-Location $repoPath

# Backup if requested
if ($BackupFirst) {
    Write-Host "Creating backup of .git directory..." -ForegroundColor Yellow
    $backupPath = ".git_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item -Path .git -Destination $backupPath -Recurse -Force
    Write-Host "Backup created at: $backupPath" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Step 1: Checking repository status..." -ForegroundColor Yellow
$issues = @()

# Check for corruption
$fsckResult = git fsck --full 2>&1
$corruptionFound = $fsckResult | Select-String "error:|missing|broken"
if ($corruptionFound) {
    $issues += "Corruption detected"
    Write-Host "  - Corruption found:" -ForegroundColor Red
    $corruptionFound | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
}

Write-Host ""
Write-Host "Step 2: Attempting repairs..." -ForegroundColor Yellow

# 1. Try to recover missing objects from origin
Write-Host "  Fetching missing objects from origin..." -ForegroundColor Cyan
git fetch origin --all 2>&1 | Out-Null
git fetch origin --tags 2>&1 | Out-Null

# 2. Clean up loose objects
Write-Host "  Cleaning loose objects..." -ForegroundColor Cyan
git prune 2>&1 | Out-Null

# 3. Remove corrupted refs
Write-Host "  Cleaning corrupted refs..." -ForegroundColor Cyan
$badRefs = git for-each-ref --format='%(refname)' | ForEach-Object {
    $ref = $_
    git show-ref --verify $ref 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Removing bad ref: $ref" -ForegroundColor Gray
        git update-ref -d $ref 2>&1 | Out-Null
    }
}

# 4. Clean up reflog
Write-Host "  Cleaning reflog..." -ForegroundColor Cyan
git reflog expire --expire=now --all 2>&1 | Out-Null

# 5. Try garbage collection with different options
Write-Host "  Running garbage collection..." -ForegroundColor Cyan
$gcResult = git gc --auto 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "    Standard gc failed, trying aggressive mode..." -ForegroundColor Yellow
    git gc --aggressive --prune=now 2>&1 | Out-Null
}

# 6. If still having issues, try to rebuild
if ($Aggressive) {
    Write-Host ""
    Write-Host "Step 3: Aggressive repair mode..." -ForegroundColor Yellow
    
    # Remove problematic directories
    $problemDirs = @(
        ".git/refs/remotes/origin/fix",
        ".git/logs/refs/remotes/origin/fix"
    )
    
    foreach ($dir in $problemDirs) {
        if (Test-Path $dir) {
            Write-Host "  Removing problematic directory: $dir" -ForegroundColor Gray
            Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    
    # Rebuild index
    Write-Host "  Rebuilding index..." -ForegroundColor Cyan
    Remove-Item .git/index -Force -ErrorAction SilentlyContinue
    git reset 2>&1 | Out-Null
    
    # Re-fetch all branches
    Write-Host "  Re-fetching all branches..." -ForegroundColor Cyan
    git remote prune origin 2>&1 | Out-Null
    git fetch origin 2>&1 | Out-Null
}

Write-Host ""
Write-Host "Step 4: Final verification..." -ForegroundColor Yellow
$finalCheck = git fsck --full 2>&1
$stillHasIssues = $finalCheck | Select-String "error:|missing|broken"

if ($stillHasIssues) {
    Write-Host "  Some issues remain:" -ForegroundColor Yellow
    $stillHasIssues | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    
    Write-Host ""
    Write-Host "Recommended next steps:" -ForegroundColor Cyan
    Write-Host "  1. Try aggressive mode: .\Repair-Git.ps1 -Aggressive" -ForegroundColor White
    Write-Host "  2. Or clone a fresh copy:" -ForegroundColor White
    Write-Host "     cd .." -ForegroundColor Gray
    Write-Host "     git clone <your-repo-url> kha_fresh" -ForegroundColor Gray
    Write-Host "     Copy your local changes from kha to kha_fresh" -ForegroundColor Gray
} else {
    Write-Host "  Repository appears healthy!" -ForegroundColor Green
    
    # Show current status
    Write-Host ""
    Write-Host "Current repository status:" -ForegroundColor Cyan
    git status --short
}

Write-Host ""
Write-Host "Repair process complete!" -ForegroundColor Green
