# D:\Dev\kha\tori_ui_svelte\Start-Dev.ps1
# Start iRis dev server with proper environment

Write-Host "=== Starting iRis Dev Server ===" -ForegroundColor Cyan
Write-Host ""

# Set location
Set-Location D:\Dev\kha\tori_ui_svelte

# Set environment variables
$env:IRIS_USE_MOCKS = "1"
$env:IRIS_ALLOW_UNAUTH = "1"

# Create uploads directory if needed
if (-not(Test-Path .\var\uploads)) {
    Write-Host "Creating uploads directory..." -ForegroundColor Yellow
    New-Item -Path .\var\uploads -ItemType Directory -Force | Out-Null
}

Write-Host "Starting dev server on http://localhost:5173" -ForegroundColor Green
Write-Host "Root will redirect to /hologram" -ForegroundColor Yellow
Write-Host ""
Write-Host "Available routes:" -ForegroundColor Cyan
Write-Host "  /hologram        - iRis HUD" -ForegroundColor White
Write-Host "  /device/demo     - Device capabilities demo" -ForegroundColor White
Write-Host "  /device/matrix   - Device matrix (use mobile UA)" -ForegroundColor White
Write-Host "  /api/health      - Health check endpoint" -ForegroundColor White
Write-Host "  /api/penrose/*   - Penrose proxy (if running on :7401)" -ForegroundColor White
Write-Host ""

# Start dev server
pnpm dev