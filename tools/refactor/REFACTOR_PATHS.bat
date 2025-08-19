@echo off
REM Quick batch file to run the refactoring

echo Starting path refactoring...
echo.

REM First do a dry run to see what will change
echo STEP 1: Dry run to preview changes
echo =====================================
python tools\refactor\mass_refactor_simple.py --dry-run

echo.
echo =====================================
echo.

set /p CONTINUE="Do you want to proceed with the actual refactoring? (y/n): "
if /i "%CONTINUE%" NEQ "y" (
    echo Refactoring cancelled.
    exit /b
)

echo.
set /p BACKUP="Enter backup directory path (or press Enter to skip backup): "

if "%BACKUP%"=="" (
    echo Running refactoring WITHOUT backup...
    python tools\refactor\mass_refactor_simple.py
) else (
    echo Running refactoring with backup to: %BACKUP%
    python tools\refactor\mass_refactor_simple.py --backup-dir "%BACKUP%"
)

echo.
echo Refactoring complete!
pause
