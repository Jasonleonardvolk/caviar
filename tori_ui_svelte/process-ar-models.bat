@echo off
echo =====================================
echo   Processing AR Models
echo =====================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

REM Check status only
if "%1"=="check" (
    powershell -ExecutionPolicy Bypass -File "tools\assets\Process-ARModels.ps1" -CheckOnly
    goto :end
)

REM Process models
powershell -ExecutionPolicy Bypass -File "tools\assets\Process-ARModels.ps1"

:end
pause