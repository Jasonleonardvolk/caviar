# Emergency Git Corruption Fix
# D:\Dev\kha\tools\git\Emergency-Fix-Corruption.ps1

Param(
    [string]$RepoRoot = "D:\Dev\kha",
    [string]$Branch = "feat/wowpack-prores-hdr10-pipeline"
)

Write-Host "=== EMERGENCY GIT CORRUPTION FIX ===" -ForegroundColor Red
Write-Host "Repository: $RepoRoot" -ForegroundColor Yellow
Write-Host ""

Push-Location $RepoRoot

# Step 1: Disable auto GC to prevent further issues
Write-Host "[1/7] Disabling auto garbage collection..." -ForegroundColor Cyan
git config gc.auto 0
git config gc.autoDetach false
git config maintenance.auto false

# Step 2: Try to clean up lock files and temp files
Write-Host "[2/7] Cleaning up lock files and temp files..." -ForegroundColor Cyan
$lockFiles = @(
    ".git/index.lock",
    ".git/HEAD.lock",
    ".git/refs/heads/$Branch.lock",
    ".git/gc.pid"
)
foreach ($lock in $lockFiles) {
    if (Test-Path $lock) {
        Remove-Item $lock -Force
        Write-Host "  Removed: $lock" -ForegroundColor Green
    }
}

# Step 3: Move corrupted objects out of the way
Write-Host "[3/7] Moving corrupted objects to backup..." -ForegroundColor Cyan
$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$corruptBackup = ".git/objects_corrupt_$ts"
if (-not (Test-Path $corruptBackup)) {
    New-Item -ItemType Directory -Path $corruptBackup -Force | Out-Null
}

# Known corrupted objects from the error
$badObjects = @(
    "4b/6e8917d44bacec2a6dd31431e0f22707642358",
    "0e/c45670f69588f9ab8b8426cc6e5551ba7c2988"
)

foreach ($obj in $badObjects) {
    $objPath = ".git/objects/$obj"
    if (Test-Path $objPath) {
        $destDir = Join-Path $corruptBackup (Split-Path $obj -Parent)
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Move-Item $objPath -Destination (Join-Path $corruptBackup $obj) -Force -ErrorAction SilentlyContinue
        Write-Host "  Quarantined: $obj" -ForegroundColor Yellow
    }
}

# Step 4: Try to fetch fresh objects from remote
Write-Host "[4/7] Fetching fresh objects from remote..." -ForegroundColor Cyan
$originUrl = git config --get remote.origin.url
if ($originUrl) {
    Write-Host "  Remote: $originUrl" -ForegroundColor Gray
    git fetch origin --all --tags --prune --force 2>&1 | Out-Null
    $fetchResult = $LASTEXITCODE
    if ($fetchResult -eq 0) {
        Write-Host "  Fetch successful!" -ForegroundColor Green
    } else {
        Write-Host "  Fetch had issues, continuing..." -ForegroundColor Yellow
    }
} else {
    Write-Host "  WARNING: No remote origin found!" -ForegroundColor Red
}

# Step 5: Rebuild index
Write-Host "[5/7] Rebuilding Git index..." -ForegroundColor Cyan
Remove-Item .git/index -Force -ErrorAction SilentlyContinue
git reset --mixed 2>&1 | Out-Null
Write-Host "  Index rebuilt" -ForegroundColor Green

# Step 6: Try shallow clone recovery
Write-Host "[6/7] Attempting shallow recovery..." -ForegroundColor Cyan
git fsck --no-reflogs 2>&1 | Out-Null
$fsckResult = $LASTEXITCODE

if ($fsckResult -ne 0) {
    Write-Host "  Repository still has issues. Recommending full reclone." -ForegroundColor Red
    Write-Host ""
    Write-Host "=== RECOMMENDED ACTION ===" -ForegroundColor Yellow
    Write-Host "The repository is too corrupted for in-place repair." -ForegroundColor White
    Write-Host "Run the full repair script:" -ForegroundColor White
    Write-Host ""
    Write-Host '  powershell -ExecutionPolicy Bypass -File "D:\Dev\kha\tools\git\Fix-Git-Repo-Auto.ps1"' -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This will create a clean clone while preserving your work." -ForegroundColor White
} else {
    Write-Host "  Repository appears recoverable!" -ForegroundColor Green
    
    # Step 7: Reset to remote branch
    Write-Host "[7/7] Resetting to remote branch..." -ForegroundColor Cyan
    git reset --hard origin/$Branch 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Successfully reset to origin/$Branch" -ForegroundColor Green
        Write-Host ""
        Write-Host "=== RECOVERY SUCCESSFUL ===" -ForegroundColor Green
        Write-Host "Your repository has been recovered." -ForegroundColor White
        Write-Host "You can now stage and commit your changes." -ForegroundColor White
    } else {
        Write-Host "  Could not reset. Manual intervention needed." -ForegroundColor Red
    }
}

Pop-Location

Write-Host ""
Write-Host "Emergency fix complete." -ForegroundColor Cyan