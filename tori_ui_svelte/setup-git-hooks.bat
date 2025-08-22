@echo off
REM Quick Git workflow setup for iRis

echo =====================================
echo     iRis Git Workflow Setup
echo =====================================
echo.
echo This will configure Git hooks for the iRis project
echo.
pause

cd /d D:\Dev\kha\tori_ui_svelte

powershell -ExecutionPolicy Bypass -File "tools\git\Setup-GitHooks.ps1"

echo.
echo Setup complete! Git hooks are now active.
echo.
echo Every 'git push' will now run verification checks.
echo.
pause