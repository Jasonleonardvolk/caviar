@echo off
REM Quick API test script

echo ========================================
echo Testing TORI API Startup
echo ========================================
echo.

REM First run troubleshooting
echo Running diagnostics...
python troubleshoot_api.py
echo.

echo ========================================
echo Press any key to try starting API manually...
pause >nul

REM Try to start the API server directly
echo.
echo Starting enhanced_launcher.py...
python enhanced_launcher.py

pause
