@echo off
echo ========================================
echo TORI Frontend Direct Test
echo ========================================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte"

echo Checking environment...
if not exist "package.json" (
    echo ERROR: package.json not found!
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo WARNING: node_modules not found - running npm install...
    call npm install
)

echo.
echo Starting frontend directly...
echo.

REM Set environment variables
set PORT=5173
set HOST=0.0.0.0
set NODE_OPTIONS=--max-old-space-size=4096
set FORCE_COLOR=0
set VITE_ENABLE_CONCEPT_MESH=true

echo Environment set:
echo   PORT=%PORT%
echo   HOST=%HOST%
echo   NODE_OPTIONS=%NODE_OPTIONS%
echo.

echo Starting npm run dev...
echo ========================================
call npm run dev

pause
