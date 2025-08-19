# Watch-And-Push.ps1
# FULLY AUTOMATED - Watches for changes and pushes every 15 seconds
param(
    [int]$IntervalSeconds = 15
)

Write-Host "AUTO-PUSH ACTIVE - Pushing every $IntervalSeconds seconds" -ForegroundColor Magenta
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow

while ($true) {
    $changes = git status --porcelain
    if ($changes) {
        $time = Get-Date -Format "HH:mm:ss"
        Write-Host "[$time] Pushing changes..." -ForegroundColor Cyan
        git add -A
        git commit -m "auto-sync $time"
        git push
        Write-Host "[$time] Pushed!" -ForegroundColor Green
    }
    Start-Sleep -Seconds $IntervalSeconds
}