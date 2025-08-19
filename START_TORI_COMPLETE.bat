@echo off
echo ========================================
echo Complete TORI System Startup
echo ========================================
echo.

REM Check if backend is already running on port 8002
netstat -an | findstr :8002 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo Backend already running on port 8002!
) else (
    echo Starting backend API on port 8002...
    start "TORI Backend" cmd /k "START_BACKEND_8002.bat"
    echo Waiting for backend to initialize...
    timeout /t 5 /nobreak >nul
)

echo.
echo Starting frontend...
cd tori_ui_svelte
npm run dev -- --host

REM This will keep the frontend running in this window
REM To stop everything, close both terminal windows
