@echo off
REM ONE-BUTTON IRIS RELEASE GATE
REM Double-click to run all ship criteria checks

cd /d D:\Dev\kha

echo ========================================
echo     IRIS ONE-BUTTON RELEASE GATE
echo ========================================
echo.
echo Running all ship criteria checks...
echo.

powershell -ExecutionPolicy Bypass -File .\tools\release\IrisOneButton.ps1

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo        ALL CHECKS PASSED!
    echo         READY TO SHIP!
    echo ========================================
    color 0A
) else (
    echo.
    echo ========================================
    echo        CHECKS FAILED!
    echo     See logs above for details
    echo ========================================
    color 0C
)

echo.
pause
