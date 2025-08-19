@echo off
title TORI System Launcher
color 0A

echo ========================================
echo     TORI SYSTEM LAUNCHER
echo     Starting all components...
echo ========================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha

echo [1/3] Running pre-flight checks...
python pre_flight_check.py
if %errorlevel% neq 0 (
    echo.
    echo Pre-flight checks failed!
    echo Running emergency fix...
    python tori_emergency_fix.py
    echo.
    echo Retrying pre-flight checks...
    python pre_flight_check.py
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: System not ready. Please fix issues manually.
        pause
        exit /b 1
    )
)

echo.
echo [2/3] Starting isolated components...
start "TORI Components" cmd /k python isolated_startup.py

echo.
echo [3/3] Waiting for services to initialize...
timeout /t 20 /nobreak > nul

echo.
echo ========================================
echo     TORI SYSTEM READY!
echo ========================================
echo.
echo Access Points:
echo   - Frontend: http://localhost:5173
echo   - API Docs: http://localhost:8002/docs
echo   - Health: http://localhost:8002/api/health
echo.
echo To monitor health, run: python health_monitor.py
echo To stop TORI, run: python shutdown_tori.py
echo.
echo Opening frontend in browser...
timeout /t 3 /nobreak > nul
start http://localhost:5173

echo.
echo This window can be closed safely.
pause
