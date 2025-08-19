@echo off
REM Build Penrose Engine Rust Extension
REM Batch script wrapper for PowerShell build script

echo ========================================
echo   Penrose Engine Rust Build Script
echo ========================================
echo.

REM Check if PowerShell script exists
if not exist "%~dp0build_penrose.ps1" (
    echo ERROR: build_penrose.ps1 not found!
    pause
    exit /b 1
)

REM Parse command line arguments
set ARGS=
if "%1"=="clean" set ARGS=-Clean
if "%1"=="dev" set ARGS=-Development
if "%1"=="test" set ARGS=-Test
if "%1"=="clean-test" set ARGS=-Clean -Test
if "%1"=="clean-dev" set ARGS=-Clean -Development

REM Run PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0build_penrose.ps1" %ARGS%

if %errorlevel% neq 0 (
    echo.
    echo Build failed!
    pause
    exit /b %errorlevel%
)

echo.
echo Build completed successfully!
echo.
pause
