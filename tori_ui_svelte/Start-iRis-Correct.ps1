# D:\Dev\kha\tori_ui_svelte\Start-iRis-Correct.ps1
# CORRECT way to start iRis dev server on port 5173

Write-Host @"
==================================================
           STARTING iRis DEV SERVER
           Port: 5173
           Dir:  tori_ui_svelte
==================================================
"@ -ForegroundColor Cyan

# MUST be in tori_ui_svelte directory
Set-Location D:\Dev\kha\tori_ui_svelte

# Check if port 5173 is already taken
$existing = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[ERROR] Port 5173 is already in use!" -ForegroundColor Red
    Write-Host "Run: D:\Dev\kha\Kill-Wrong-Ports.ps1" -ForegroundColor Yellow
    exit 1
}

# Set environment
$env:IRIS_USE_MOCKS = "1"
$env:IRIS_ALLOW_UNAUTH = "1"

Write-Host ""
Write-Host "Starting iRis dev server..." -ForegroundColor Green
Write-Host "URLs:" -ForegroundColor White
Write-Host "  http://localhost:5173/hologram       - iRis HUD" -ForegroundColor Gray
Write-Host "  http://localhost:5173/device/demo    - Device Demo" -ForegroundColor Gray
Write-Host "  http://localhost:5173/api/penrose/*  - Penrose Proxy" -ForegroundColor Gray
Write-Host ""
Write-Host "DO NOT append uvicorn commands here!" -ForegroundColor Yellow
Write-Host ""

# Start Vite ONLY - no uvicorn arguments!
pnpm dev