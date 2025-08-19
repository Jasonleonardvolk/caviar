@echo off
REM =======================================================
REM ELFIN Error Documentation Tools Runner
REM =======================================================

if "%1"=="" (
    echo ELFIN Error Documentation Tools
    echo Usage: run_error_tools.bat [command] [args]
    echo.
    echo Commands:
    echo   new CODE     - Create a new error documentation file
    echo                  Example: run_error_tools.bat new LYAP_003
    echo.
    echo   verify       - Verify that error documentation exists
    echo                  Example: run_error_tools.bat verify LYAP_001 LYAP_002
    echo.
    echo   build        - Build error documentation site
    echo                  Example: run_error_tools.bat build
    echo.
    echo   explain CODE - Show explanation for error code
    echo                  Example: run_error_tools.bat explain LYAP_001
    goto :end
)

set COMMAND=%1

if "%COMMAND%"=="new" (
    echo Creating new error documentation for %2...
    python -m alan_backend.elfin.cli errors new %2
    goto :end
)

if "%COMMAND%"=="verify" (
    echo Verifying error documentation...
    python -m alan_backend.elfin.cli errors verify %2 %3 %4 %5 %6 %7 %8 %9
    goto :end
)

if "%COMMAND%"=="build" (
    echo Building error documentation...
    python -m alan_backend.elfin.cli errors build
    goto :end
)

if "%COMMAND%"=="explain" (
    echo Showing explanation for %2...
    python -m alan_backend.elfin.cli verify --explain %2
    goto :end
)

echo Unknown command: %COMMAND%
echo Run 'run_error_tools.bat' without arguments for usage information.

:end
