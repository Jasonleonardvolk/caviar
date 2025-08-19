@echo off
REM ONE-CLICK REFACTORING - Just double-click this file!

cd /d D:\Dev\kha

echo Starting path refactoring...
echo This will process ~50,000 files in your 23GB repository.
echo.
echo Press Ctrl+C anytime to stop (you can resume with CONTINUE_REFACTOR.bat)
echo.

python tools\refactor\refactor_fast.py --backup-dir "D:\Backups\KhaRefactor_%date:~-4,4%%date:~-10,2%%date:~-7,2%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo REFACTORING INTERRUPTED!
    echo To continue, run: CONTINUE_REFACTOR.bat
    echo ========================================
) else (
    echo.
    echo ========================================
    echo REFACTORING COMPLETE!
    echo Check tools\refactor\ for the log file
    echo ========================================
)
pause
