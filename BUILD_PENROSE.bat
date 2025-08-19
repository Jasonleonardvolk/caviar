@echo off
echo.
echo ============================================
echo    Building Penrose Engine (Rust)
echo ============================================
echo.

REM Run the PowerShell script
powershell.exe -ExecutionPolicy Bypass -File "%~dp0BUILD_PENROSE.ps1"