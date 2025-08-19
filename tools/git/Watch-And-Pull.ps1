# Watch-And-Pull.ps1
# RECEIVING END - Auto-pulls changes from GitHub every 15 seconds
# Your friend runs this to stay synced with your pushes
param(
    [int]$IntervalSeconds = 120,
    [switch]$ShowDiff
)

Write-Host "AUTO-PULL ACTIVE - Checking for updates every 2 minutes" -ForegroundColor Magenta
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "Repository: $(git remote get-url origin)" -ForegroundColor Gray

$lastCommit = git rev-parse HEAD

while ($true) {
    # Fetch without merging first to check for changes
    git fetch origin main --quiet
    
    $remoteCommit = git rev-parse origin/main
    $localCommit = git rev-parse HEAD
    
    if ($remoteCommit -ne $localCommit) {
        $time = Get-Date -Format "HH:mm:ss"
        Write-Host "[$time] New changes detected!" -ForegroundColor Cyan
        
        if ($ShowDiff) {
            Write-Host "Changes:" -ForegroundColor Yellow
            git diff HEAD..origin/main --stat
        }
        
        # Pull the changes
        git pull origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$time] ✓ Pulled latest changes" -ForegroundColor Green
            
            # Show what changed
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