@echo off
echo ===============================================
echo TORI CACHE CLEAR AND RESTART SCRIPT
echo ===============================================
echo.
echo This script will:
echo 1. Stop all TORI services
echo 2. Clear ALL caches
echo 3. Restart everything fresh
echo.
echo MAKE SURE YOU HAVE STOPPED THE DEV SERVER (Ctrl+C)
echo.
pause

echo.
echo Step 1: Clearing Svelte/Vite caches...
cd /d "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte"

if exist ".svelte-kit" (
    echo Removing .svelte-kit...
    rmdir /s /q ".svelte-kit"
)

if exist "node_modules\.vite" (
    echo Removing node_modules\.vite...
    rmdir /s /q "node_modules\.vite"
)

if exist ".vite" (
    echo Removing .vite...
    rmdir /s /q ".vite"
)

echo.
echo Step 2: Clearing browser storage...
echo Please open your browser and:
echo 1. Press F12 to open Developer Tools
echo 2. Go to Application tab
echo 3. Click "Clear site data" or manually clear:
echo    - Local Storage
echo    - Session Storage
echo    - Cookies
echo    - Service Workers
echo.
pause

echo.
echo Step 3: Rebuilding frontend...
echo Running npm install to ensure dependencies...
call npm install

echo.
echo Step 4: Starting TORI with fresh build...
cd /d "C:\Users\jason\Desktop\tori\kha"
echo.
echo Starting enhanced launcher...
python enhanced_launcher.py

echo.
echo ===============================================
echo If the launcher started successfully:
echo 1. Wait for all services to initialize
echo 2. Open http://localhost:5173 in an INCOGNITO window
echo 3. Check the browser console for errors (F12)
echo ===============================================
pause
