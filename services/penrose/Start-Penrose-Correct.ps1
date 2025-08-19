# D:\Dev\kha\services\penrose\Start-Penrose-Correct.ps1
# CORRECT way to start Penrose API on port 7401

Write-Host @"
==================================================
           STARTING PENROSE API
           Port: 7401
           Dir:  services\penrose
==================================================
"@ -ForegroundColor Cyan

# MUST be in Penrose directory
Set-Location D:\Dev\kha\services\penrose

# Check if port 7401 is already taken
$existing = Get-NetTCPConnection -LocalPort 7401 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[ERROR] Port 7401 is already in use!" -ForegroundColor Red
    Write-Host "Run: D:\Dev\kha\Kill-Wrong-Ports.ps1" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating Python virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Starting Penrose API..." -ForegroundColor Green
Write-Host "URL: http://127.0.0.1:7401/docs" -ForegroundColor White
Write-Host ""

# Start Uvicorn - DO NOT mix with Vite commands!
uvicorn main:app --host 0.0.0.0 --port 7401