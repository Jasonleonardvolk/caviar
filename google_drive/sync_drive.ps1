# Google Drive Sync for Tori Project
# PowerShell script for synchronization

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Google Drive Sync for Tori Project" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
Set-Location "C:\Users\jason\Desktop\tori\kha"

# Check if virtual environment exists and activate it
$venvPath = ".venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venvPath
}

# Install dependencies if needed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$dependencies = @(
    "google-api-python-client",
    "google-auth-httplib2",
    "google-auth-oauthlib"
)

foreach ($dep in $dependencies) {
    pip show $dep 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing $dep..." -ForegroundColor Yellow
        pip install $dep
    }
}

Write-Host ""
Write-Host "Starting synchronization..." -ForegroundColor Green
Write-Host ""

# Run the sync script
python google_drive\drive_sync.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Sync completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Sync failed! Check the error messages above." -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
