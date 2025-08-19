# D:\Dev\kha\services\penrose\start-penrose.ps1
# Start Penrose assist API

Write-Host "=== Starting Penrose API Service ===" -ForegroundColor Cyan
Write-Host ""

# Set location
Set-Location D:\Dev\kha\services\penrose

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Starting Penrose API on http://127.0.0.1:7401" -ForegroundColor Green
Write-Host ""
Write-Host "Available endpoints:" -ForegroundColor Cyan
Write-Host "  http://localhost:7401/docs     - API documentation" -ForegroundColor White
Write-Host "  http://localhost:7401/api/*    - API endpoints" -ForegroundColor White
Write-Host ""
Write-Host "iRis proxy access (when dev/prod running):" -ForegroundColor Yellow
Write-Host "  http://localhost:5173/api/penrose/*  - Dev proxy" -ForegroundColor White
Write-Host "  http://localhost:3000/api/penrose/*  - Prod proxy" -ForegroundColor White
Write-Host ""

# Start Uvicorn server
uvicorn main:app --host 0.0.0.0 --port 7401