@echo off
REM Quick check for any absolute paths that might have crept back in

cd /d D:\Dev\kha

echo ========================================
echo ABSOLUTE PATH CHECK
echo ========================================
echo.

node tools\runtime\preflight.mjs

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo FAILED: Absolute paths detected!
    echo ========================================
    echo.
    echo To fix these, run:
    echo   python tools\refactor\refactor_continue.py
    echo.
) else (
    echo.
    echo ========================================
    echo PASSED: No absolute paths found
    echo ========================================
)

pause
