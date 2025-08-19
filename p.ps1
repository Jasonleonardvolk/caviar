# p.ps1
# ULTRA SHORT push command - just type 'p' and Enter!
param([string]$m = "sync")
git add -A; git commit -m $m; git push
Write-Host "Pushed!" -ForegroundColor Green