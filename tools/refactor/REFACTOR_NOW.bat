@echo off
REM Direct refactoring for large repositories - no dry run preview

echo ========================================
echo PATH REFACTORING TOOL
echo ========================================
echo.
echo This will replace all occurrences of:
echo   C:\Users\jason\Desktop\tori\kha
echo.
echo With:
echo   {PROJECT_ROOT} in Python files
echo   ${IRIS_ROOT} in other files
echo.
echo ========================================
echo.

set /p BACKUP="Enter backup directory path (or press Enter to skip backup): "

if "%BACKUP%"=="" (
    echo.
    set /p CONFIRM="WARNING: No backup directory specified. Continue anyway? (y/n): "
    if /i "%CONFIRM%" NEQ "y" (
        echo Refactoring cancelled.
        exit /b
    )
    echo.
    echo Starting refactoring WITHOUT backup...
    echo This may take a while for 23GB of data...
    echo.
    python tools\refactor\mass_refactor_simple.py
) else (
    echo.
    echo Starting refactoring with backup to: %BACKUP%
    echo This may take a while for 23GB of data...
    echo.
    python tools\refactor\mass_refactor_simple.py --backup-dir "%BACKUP%"
)

echo.
echo ========================================
echo Refactoring complete!
echo Check tools\refactor\ for logs
echo ========================================
pause
