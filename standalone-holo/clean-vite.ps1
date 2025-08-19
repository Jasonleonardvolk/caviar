# Clean Vite cache
Write-Host "Cleaning Vite cache..." -ForegroundColor Yellow

# Kill any Node processes
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Remove .vite cache from node_modules
$vitePath = ".\node_modules\.vite"
if (Test-Path $vitePath) {
    Remove-Item -Path $vitePath -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleared node_modules/.vite" -ForegroundColor Green
}

# Remove local .vite-cache
$localCache = ".\.vite-cache"
if (Test-Path $localCache) {
    Remove-Item -Path $localCache -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleared .vite-cache" -ForegroundColor Green
}

Write-Host "Vite cache cleared!" -ForegroundColor Green
Write-Host "You can now run: npm run dev" -ForegroundColor Cyan