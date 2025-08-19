# Auto-Commit-Push.ps1
# Commits and pushes EVERYTHING instantly with one command
param(
    [string]$m = "sync"  # Short message parameter
)

$timestamp = Get-Date -Format "HH:mm"
$files = git status --porcelain

if ($files) {
    Write-Host "Pushing changes..." -ForegroundColor Cyan
    git add -A
    git commit -m "$m - $timestamp"
    git push
    Write-Host "✓ Pushed to GitHub! Your friend can pull now." -ForegroundColor Green
} else {
    Write-Host "No changes to push" -ForegroundColor Yellow
}
