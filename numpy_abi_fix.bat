@echo off
REM ============================================
REM Numpy ABI Fix - Windows Batch Launcher
REM ============================================
REM This batch file provides an easy way to run the numpy ABI fix
REM Author: Enhanced Assistant
REM Date: 2025-08-06

setlocal enabledelayedexpansion

echo.
echo ============================================
echo    NUMPY ABI COMPATIBILITY FIX LAUNCHER
echo ============================================
echo.

REM Check if we're in the correct directory
if not exist "pyproject.toml" (
    echo ERROR: pyproject.toml not found!
    echo Please run this script from your project root directory.
    echo Expected location: C:\Users\jason\Desktop\tori\kha
    echo.
    pause
    exit /b 1
)

echo Current directory: %CD%
echo.

:menu
echo Please select an option:
echo.
echo   1. Run Full Fix (Recommended for ABI errors)
echo   2. Verify Current Environment
echo   3. Update pyproject.toml (Pin numpy version)
echo   4. Quick Clean (Remove caches only)
echo   5. Launch Application
echo   6. Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto full_fix
if "%choice%"=="2" goto verify
if "%choice%"=="3" goto update_project
if "%choice%"=="4" goto quick_clean
if "%choice%"=="5" goto launch_app
if "%choice%"=="6" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:full_fix
echo.
echo ============================================
echo Running Full Numpy ABI Fix...
echo ============================================
echo.
echo This will:
echo   - Backup current environment
echo   - Remove virtual environment
echo   - Clear all caches
echo   - Reinstall all packages
echo.
echo This process may take 5-10 minutes.
echo.

set /p confirm="Are you sure you want to continue? (y/n): "
if /i not "%confirm%"=="y" goto menu

echo.
echo Starting fix...
powershell -ExecutionPolicy Bypass -File fix_numpy_abi.ps1

if %errorlevel% equ 0 (
    echo.
    echo Fix completed successfully!
    echo.
    set /p verify_now="Would you like to verify the environment now? (y/n): "
    if /i "!verify_now!"=="y" goto verify
) else (
    echo.
    echo Fix encountered issues. Please check the logs.
)

echo.
pause
goto menu

:verify
echo.
echo ============================================
echo Verifying Environment...
echo ============================================
echo.

poetry run python verify_numpy_abi.py

echo.
pause
goto menu

:update_project
echo.
echo ============================================
echo Updating pyproject.toml...
echo ============================================
echo.

poetry run python update_numpy_pin.py

echo.
pause
goto menu

:quick_clean
echo.
echo ============================================
echo Quick Clean - Removing Caches...
echo ============================================
echo.

echo Clearing Python cache files...
for /r %%i in (__pycache__) do if exist "%%i" rd /s /q "%%i" 2>nul
del /s /q *.pyc 2>nul

echo Clearing Poetry cache...
poetry cache clear pypi --all -n

echo.
echo Cache cleared successfully!
echo.
pause
goto menu

:launch_app
echo.
echo ============================================
echo Launching Application...
echo ============================================
echo.

set /p log_to_file="Would you like to save logs to file? (y/n): "

if /i "%log_to_file%"=="y" (
    set timestamp=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
    set timestamp=!timestamp: =0!
    set logfile=logs\backend_log_!timestamp!.txt
    
    if not exist "logs" mkdir logs
    
    echo Launching with logging to: !logfile!
    echo.
    poetry run python enhanced_launcher.py --api full --require-penrose --enable-hologram --hologram-audio 2>&1 | powershell -Command "Tee-Object -FilePath '!logfile!'"
) else (
    echo Launching without file logging...
    echo.
    poetry run python enhanced_launcher.py --api full --require-penrose --enable-hologram --hologram-audio
)

echo.
echo Application terminated.
pause
goto menu

:end
echo.
echo Goodbye!
echo.
exit /b 0