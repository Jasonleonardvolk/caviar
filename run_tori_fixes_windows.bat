@echo off
REM Windows Batch script to run TORI fixes and verification

echo ===================================================
echo TORI Fix and Verification Script for Windows
echo ===================================================

REM Step 1: Apply fixes
echo.
echo Applying fixes...
python fix_tori_automatic_v3.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Fix script failed!
    pause
    exit /b 1
)

REM Step 2: Run unit tests (optional)
if exist "tests\test_launch_order.py" (
    echo.
    echo Running unit tests...
    pytest tests\test_launch_order.py
)

REM Step 3: Start TORI in new window
echo.
echo Starting TORI in new window...
start "TORI Server" /min cmd /c "poetry run python enhanced_launcher.py"

REM Step 4: Wait for startup
echo Waiting 15 seconds for TORI to start...
timeout /t 15 /nobreak > nul

REM Step 5: Verify
echo.
echo Running verification...
python verify_tori_fixes_v3.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================
    echo SUCCESS! TORI is running!
    echo.
    echo Open your browser to: http://localhost:5173
    echo API docs at: http://localhost:8002/docs
    echo ===================================================
    echo.
    echo Press any key to stop TORI...
    pause > nul
    
    REM Kill TORI processes
    taskkill /FI "WINDOWTITLE eq TORI Server*" /T /F > nul 2>&1
    echo TORI stopped.
) else (
    echo.
    echo ERROR: Verification failed!
    echo Check the logs above for details.
    pause
)
