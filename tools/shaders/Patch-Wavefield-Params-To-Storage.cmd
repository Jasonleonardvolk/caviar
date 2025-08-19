@echo off
REM Patch-Wavefield-Params-To-Storage.cmd
REM Launcher for converting uniform buffers to storage buffers to fix array stride issues

echo ============================================
echo Wavefield Params Uniform to Storage Patcher
echo ============================================
echo.

REM Navigate to script directory
cd /d "%~dp0"

echo Options:
echo   1. Dry run (preview changes)
echo   2. Apply wavefield_params fix only
echo   3. Apply both wavefield_params AND osc_data fixes
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo Running dry run...
    powershell -ExecutionPolicy Bypass -File "Patch-Wavefield-Params-To-Storage.ps1"
) else if "%choice%"=="2" (
    echo Applying wavefield_params fix...
    powershell -ExecutionPolicy Bypass -File "Patch-Wavefield-Params-To-Storage.ps1" -Apply
) else if "%choice%"=="3" (
    echo Applying wavefield_params AND osc_data fixes...
    powershell -ExecutionPolicy Bypass -File "Patch-Wavefield-Params-To-Storage.ps1" -Apply -IncludeOscData
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

echo.
echo ============================================
echo Operation complete!
echo ============================================
pause
