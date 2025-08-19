# Commit-WOW-Pack.ps1
# Commits the entire WOW Pack v1 with proper message

Write-Host "`n=== COMMITTING WOW PACK v1 ===" -ForegroundColor Magenta

Set-Location D:\Dev\kha

# Show what we're committing
Write-Host "`nFiles to commit:" -ForegroundColor Cyan
git status --short

# Stage everything
git add -A

# Commit with the exact message
git commit -m "feat: WOW Pack v1 (show modes + launcher) - $(Get-Date -Format 'yyyy-MM-dd')"

# Push to caviar
git push

Write-Host "`n✅ WOW Pack v1 pushed to GitHub!" -ForegroundColor Green
Write-Host "URL: https://github.com/Jasonleonardvolk/caviar" -ForegroundColor Cyan