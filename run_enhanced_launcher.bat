@echo off
REM Enhanced TORI Launcher Batch Script v3.0
REM Bulletproof Edition

echo ========================================
echo   ENHANCED TORI LAUNCHER v3.0
echo   BULLETPROOF EDITION
echo ========================================
echo.

REM Check if Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Get Python version
echo Python found at:
where python
python --version
echo.

REM Set working directory to script location
cd /d "%~dp0"
echo Working directory: %cd%
echo.

REM Parse command line arguments
set PORT=8002
set API_ONLY=
set DEBUG=

:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--api-only" (
    set API_ONLY=--api-only
    shift
    goto parse_args
)
if /i "%~1"=="--debug" (
    set DEBUG=--debug
    shift
    goto parse_args
)
shift
goto parse_args
:end_parse

echo Configuration:
echo   API Port: %PORT%
if defined API_ONLY echo   Mode: API Only
if defined DEBUG echo   Debug: Enabled
echo.

REM Check if port is already in use
netstat -an | findstr ":%PORT% " >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: Port %PORT% appears to be in use
    echo.
    set /p CONTINUE="Continue anyway? (y/N): "
    if /i not "%CONTINUE%"=="y" (
        echo Aborted.
        pause
        exit /b 1
    )
)

echo Starting Enhanced TORI Launcher...
echo.

REM Launch the Python script
python enhanced_launcher.py --port %PORT% %API_ONLY% %DEBUG%

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Launcher failed with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo TORI Launcher has terminated.
pause
