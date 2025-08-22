# D:\Dev\kha\tori_ui_svelte\Start-Prod.ps1
# Build and start iRis production SSR

Write-Host "=== Starting iRis Production Build ===" -ForegroundColor Cyan
Write-Host ""

# Set location
Set-Location D:\Dev\kha\tori_ui_svelte

# Ensure dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
pnpm install

# Build the project
Write-Host ""
Write-Host "Building production bundle..." -ForegroundColor Yellow
pnpm run build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting production server on http://localhost:3000" -ForegroundColor Green
Write-Host "Root will redirect to /hologram" -ForegroundColor Yellow
Write-Host ""

# Start production server
node .\build\index.js