param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "QUICK TEST - IRIS PREVIEW" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nStarting IRIS preview server..." -ForegroundColor Green
Write-Host "This serves the production build locally" -ForegroundColor Gray

Set-Location "tori_ui_svelte"

Write-Host "`nThe app will be available at:" -ForegroundColor Yellow
Write-Host "  http://localhost:4173" -ForegroundColor Cyan
Write-Host "`nIf the browser doesn't open automatically:" -ForegroundColor Gray
Write-Host "  1. Open your browser" -ForegroundColor White
Write-Host "  2. Go to http://localhost:4173" -ForegroundColor White
Write-Host "`nTo stop the server: Press Ctrl+C" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

npm run preview