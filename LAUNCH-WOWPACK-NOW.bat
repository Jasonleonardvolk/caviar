@echo off
cls
echo.
echo ===============================================
echo     LAUNCH WOW PACK DEMO - QUICK SETUP
echo ===============================================
echo.

REM Run the PowerShell setup script
echo [1/3] Setting up WOW Pack outputs...
powershell -ExecutionPolicy Bypass -File "D:\Dev\kha\tools\release\Setup-WowPack-Outputs.ps1"

echo.
echo [2/3] Instructions for dev server...
echo.
echo If the dev server is not running, please:
echo   1. Open a new terminal
echo   2. cd D:\Dev\kha\frontend
echo   3. pnpm dev
echo.

echo [3/3] Ready to demo!
echo.
echo ===============================================
echo     OPEN: http://localhost:5173/hologram
echo ===============================================
echo.
echo You will see:
echo   - Hologram canvas (top)
echo   - Recorder controls (middle)
echo   - WOW Pack Demo Loops (bottom) - NEW!
echo.
echo Click the video tabs to showcase:
echo   - HOLO FLUX (wave patterns)
echo   - MACH LIGHTFIELD (interference)
echo   - KINETIC LOGO PARADE (motion graphics)
echo.
echo Press any key to open the browser...
pause >nul

start http://localhost:5173/hologram