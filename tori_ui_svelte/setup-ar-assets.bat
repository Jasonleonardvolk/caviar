@echo off
color 0E
cls
echo.
echo    ==================================================
echo           iRis AR Assets - Quick Setup
echo    ==================================================
echo.
echo    This will set up the AR assets system with
echo    free 3D models for testing.
echo.
echo    ==================================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo [1/4] Installing dependencies...
echo.
call tools\assets\install-dependencies.bat
if %ERRORLEVEL% neq 0 exit /b 1

echo.
echo [2/4] Validating manifest...
echo.
node tools\assets\validate-manifest.mjs

echo.
echo [3/4] Creating placeholder files...
echo.
REM Create placeholder files so the system works even without downloads
echo {"placeholder": true} > assets\3d\luxury\sneakers\free\cgtrader_generic_sneaker\raw\placeholder.json
echo {"placeholder": true} > assets\3d\luxury\watches\free\sketchfab_lowpoly_watch\raw\placeholder.json
echo {"placeholder": true} > assets\3d\luxury\perfume\free\sketchfab_perfume_bottle\raw\placeholder.json

echo.
echo ====================================================
echo              Setup Complete!
echo ====================================================
echo.
echo Download the FREE 3D models from:
echo.
echo 1. SNEAKER:
echo    https://www.cgtrader.com/free-3d-models/sports/equipment
echo    Save to: assets\3d\luxury\sneakers\free\cgtrader_generic_sneaker\raw\
echo.
echo 2. WATCH:
echo    https://sketchfab.com/3d-models/low-poly-watch-85e3fe10d5e9411fbcd409f7eab6524f
echo    Save to: assets\3d\luxury\watches\free\sketchfab_lowpoly_watch\raw\
echo.
echo 3. PERFUME:
echo    https://sketchfab.com/3d-models/3d-perfume-bottle-for-product-visualizations-6e58cd19fb4f49829411c7f97df14f6e
echo    Save to: assets\3d\luxury\perfume\free\sketchfab_perfume_bottle\raw\
echo.
echo After downloading, optimize each model:
echo   pwsh tools\assets\ingest-3d.ps1 [raw-file] [output-dir] draco
echo.
echo Then view at: http://localhost:3000/assets
echo.
pause