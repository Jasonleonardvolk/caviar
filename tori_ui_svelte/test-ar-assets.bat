@echo off
echo =====================================
echo    Testing AR Assets System
echo =====================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

powershell -ExecutionPolicy Bypass -File "tools\assets\Test-ARAssets.ps1"

pause