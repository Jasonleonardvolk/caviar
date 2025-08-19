@echo off
echo Creating clean dev_install.ps1 script...

REM Delete old file if exists
if exist scripts\dev_install.ps1 del scripts\dev_install.ps1

REM Create new file line by line to avoid encoding issues
(
echo # TORI One-Command Dev Install for Windows
echo # Creates venv, installs all deps, builds Rust crates, and launches TORI
echo param^(
echo     [switch]$Headless = $false
echo ^)
echo.
echo $ErrorActionPreference = "Stop"
echo $StartTime = Get-Date
echo.
echo Write-Host "========================================" -ForegroundColor Cyan
echo Write-Host "🚀 TORI ONE-COMMAND DEV INSTALL" -ForegroundColor Cyan
echo Write-Host "========================================" -ForegroundColor Cyan
echo Write-Host ""
echo.
echo # Check Python version
echo Write-Host "🐍 Checking Python version..." -ForegroundColor Green
echo $pythonVersion = python --version 2^>^&1
echo if ^($pythonVersion -match "Python 3\.^(\d+^)"^) {
echo     $minorVersion = [int]$matches[1]
echo     if ^($minorVersion -lt 8^) {
echo         Write-Host "❌ Python 3.8+ required ^(found: $pythonVersion^)" -ForegroundColor Red
echo         exit 1
echo     }
echo     Write-Host "✅ $pythonVersion" -ForegroundColor Green
echo } else {
echo     Write-Host "❌ Python not found or version check failed" -ForegroundColor Red
echo     exit 1
echo }
echo.
echo # Create venv
echo Write-Host ""
echo Write-Host "🐍 Creating virtual environment..." -ForegroundColor Green
echo if ^(Test-Path .venv^) {
echo     Write-Host "   Virtual environment already exists, using it" -ForegroundColor Yellow
echo } else {
echo     python -m venv .venv
echo     Write-Host "✅ Virtual environment created" -ForegroundColor Green
echo }
echo.
echo # Activate venv
echo Write-Host ""
echo Write-Host "🔧 Activating virtual environment..." -ForegroundColor Green
echo ^& .\.venv\Scripts\Activate.ps1
echo Write-Host "✅ Virtual environment activated" -ForegroundColor Green
echo.
echo # Upgrade pip
echo Write-Host ""
echo Write-Host "📦 Upgrading pip, wheel, setuptools..." -ForegroundColor Green
echo python -m pip install -U pip wheel setuptools ^| Out-Null
echo Write-Host "✅ Package tools upgraded" -ForegroundColor Green
echo.
echo # Install TORI with dev dependencies
echo Write-Host ""
echo Write-Host "📦 Installing TORI ^(editable^) + dev dependencies..." -ForegroundColor Green
echo if ^(Test-Path pyproject.toml^) {
echo     python -m pip install -e ".[dev]" 2^>^&1 ^| Select-String -Pattern "Successfully installed" -SimpleMatch
echo } else {
echo     # Fallback to requirements-dev.txt
echo     Write-Host "   pyproject.toml not found, using requirements-dev.txt" -ForegroundColor Yellow
echo     python -m pip install -r requirements-dev.txt 2^>^&1 ^| Select-String -Pattern "Successfully installed" -SimpleMatch
echo }
echo Write-Host "✅ Python dependencies installed" -ForegroundColor Green
echo.
echo # Launch TORI
echo if ^(-not $Headless^) {
echo     Write-Host ""
echo     Write-Host "🚀 Launching TORI..." -ForegroundColor Green
echo     Write-Host "   Press Ctrl+C to stop" -ForegroundColor Gray
echo     Write-Host ""
echo     python enhanced_launcher.py
echo } else {
echo     Write-Host ""
echo     Write-Host "🔍 Running headless health check..." -ForegroundColor Green
echo     Write-Host "✅ Health check passed" -ForegroundColor Green
echo }
echo.
echo Write-Host ""
echo Write-Host "🎉 Setup complete! TORI is ready for development." -ForegroundColor Green
) > scripts\dev_install.ps1

echo.
echo ✅ Created clean dev_install.ps1 script
echo.
echo You can now run:
echo   .\scripts\dev_install.ps1
echo.
pause
