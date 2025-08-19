param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DIRECT BUILD TEST - SEE REAL ERRORS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Run npm build directly WITHOUT capturing output
Write-Host "`nRunning npm run build directly (you'll see the actual error):" -ForegroundColor Yellow
Write-Host "-------------------------------------------------------" -ForegroundColor Gray

# Don't capture - let it show directly
npm run build

Write-Host "-------------------------------------------------------" -ForegroundColor Gray
Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow

# Step 2: If that failed, try vite directly
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nNow trying vite build directly in tori_ui_svelte:" -ForegroundColor Yellow
    Write-Host "-------------------------------------------------------" -ForegroundColor Gray
    
    Set-Location tori_ui_svelte
    npx vite build
    
    Write-Host "-------------------------------------------------------" -ForegroundColor Gray
    Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
    Set-Location ..
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "TEST COMPLETE - CHECK ERRORS ABOVE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

exit 0