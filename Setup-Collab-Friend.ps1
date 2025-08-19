# Setup-Collab-Friend.ps1
# ONE-TIME SETUP for your friend/collaborator
# They run this ONCE to clone and set up auto-pull

param(
    [string]$CloneDir = "D:\Dev\caviar",
    [switch]$StartWatching
)

Write-Host @"
╔════════════════════════════════════════╗
║     CAVIAR COLLABORATION SETUP         ║
║     For Friend/Reviewer                ║
╚════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Check if repo already exists
if (Test-Path "$CloneDir\.git") {
    Write-Host "`nRepo already cloned at: $CloneDir" -ForegroundColor Yellow
    Set-Location $CloneDir
    git pull
} else {
    Write-Host "`nCloning caviar repository..." -ForegroundColor Yellow
    
    # Create parent directory if needed
    $parent = Split-Path $CloneDir -Parent
    if (!(Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    
    # Clone the repo
    git clone https://github.com/Jasonleonardvolk/caviar.git $CloneDir
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to clone! Check if repo is public or you have access." -ForegroundColor Red
        exit 1
    }
    
    Set-Location $CloneDir
    Write-Host "✓ Repository cloned successfully!" -ForegroundColor Green
}

# Create shortcuts for easy access
Write-Host "`nCreating helper scripts..." -ForegroundColor Cyan

# Create pull script
@'
# pull.ps1 - Quick pull latest changes
git pull origin main
Write-Host "✓ Pulled latest changes" -ForegroundColor Green
git log --oneline -n 5
'@ | Out-File -Encoding UTF8 "pull.ps1"

# Create status script
@'
# status.ps1 - Check sync status
$remote = git rev-parse origin/main
$local = git rev-parse HEAD
if ($remote -eq $local) {
    Write-Host "✓ Up to date with remote" -ForegroundColor Green
} else {
    Write-Host "⚠ Behind remote - run .\pull.ps1" -ForegroundColor Yellow
}
git status -sb
'@ | Out-File -Encoding UTF8 "status.ps1"

# Copy Watch-And-Pull script
@'
# Watch-And-Pull.ps1
# Auto-pulls changes every 15 seconds
param([int]$IntervalSeconds = 15, [switch]$ShowDiff)

Write-Host "AUTO-PULL ACTIVE - Checking every $IntervalSeconds seconds" -ForegroundColor Magenta
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow

$lastCommit = git rev-parse HEAD

while ($true) {
    git fetch origin main --quiet
    $remoteCommit = git rev-parse origin/main
    $localCommit = git rev-parse HEAD
    
    if ($remoteCommit -ne $localCommit) {
        $time = Get-Date -Format "HH:mm:ss"
        Write-Host "[$time] New changes detected!" -ForegroundColor Cyan
        
        if ($ShowDiff) {
            git diff HEAD..origin/main --stat
        }
        
        git pull origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$time] ✓ Pulled latest changes" -ForegroundColor Green
            $filesChanged = git diff --name-only $lastCommit HEAD
            if ($filesChanged) {
                Write-Host "Files updated:" -ForegroundColor Gray
                $filesChanged | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
            }
            $lastCommit = git rev-parse HEAD
        } else {
            Write-Host "[$time] ⚠ Pull failed - may have conflicts" -ForegroundColor Red
        }
    }
    
    Start-Sleep -Seconds $IntervalSeconds
}
'@ | Out-File -Encoding UTF8 "Watch-And-Pull.ps1"

Write-Host "`n✓ Helper scripts created:" -ForegroundColor Green
Write-Host "  .\pull.ps1          - Manual pull" -ForegroundColor Gray
Write-Host "  .\status.ps1        - Check sync status" -ForegroundColor Gray
Write-Host "  .\Watch-And-Pull.ps1 - Auto-pull every 15 seconds" -ForegroundColor Gray

Write-Host "`n=== SETUP COMPLETE ===" -ForegroundColor Green
Write-Host "Repository location: $CloneDir" -ForegroundColor Cyan

if ($StartWatching) {
    Write-Host "`nStarting auto-pull watcher..." -ForegroundColor Yellow
    & ".\Watch-And-Pull.ps1"
} else {
    Write-Host "`nTo start auto-syncing, run:" -ForegroundColor Yellow
    Write-Host "  .\Watch-And-Pull.ps1" -ForegroundColor White
    Write-Host "`nYour friend's changes will appear automatically!" -ForegroundColor Cyan
}