@echo off
echo Checking model download status...
echo.

cd /d D:\Dev\kha\tori_ui_svelte

powershell -ExecutionPolicy Bypass -File "tools\assets\Check-ModelDownloads.ps1"

pause