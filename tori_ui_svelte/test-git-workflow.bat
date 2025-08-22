@echo off
REM Test Git workflow system configuration

echo =====================================
echo    Testing Git Workflow System
echo =====================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

powershell -ExecutionPolicy Bypass -File "tools\git\Test-GitWorkflow.ps1" -Verbose

echo.
pause