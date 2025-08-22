@echo off
echo ===============================================
echo      iRis v0.1.0 - Quick Ship Setup
echo ===============================================
echo.
echo This will build and launch iRis on port 3000
echo.
pause

REM Navigate to project directory
cd /d D:\Dev\kha\tori_ui_svelte

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "tools\release\Reset-And-Ship.ps1" -UsePM2

echo.
echo ===============================================
echo iRis is now running at http://localhost:3000
echo ===============================================
pause