@echo off
echo ============================================================
echo           ITORI.IO DEMO LAUNCHER
echo ============================================================
echo.
echo Choose demo mode:
echo   1. LAN Demo (Quick, no ngrok needed)
echo   2. Public Demo (ngrok tunnel, shareable URL)
echo   3. Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting LAN demo...
    powershell -ExecutionPolicy Bypass -File "%~dp0Run-Demo-LAN.ps1"
) else if "%choice%"=="2" (
    echo.
    echo Starting public demo with ngrok...
    powershell -ExecutionPolicy Bypass -File "%~dp0Run-Demo-Ngrok.ps1"
) else if "%choice%"=="3" (
    echo Exiting...
    exit
) else (
    echo Invalid choice!
    pause
    "%~f0"
)