@echo off
echo =====================================
echo   Testing Pre-Push Hook
echo =====================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

powershell -ExecutionPolicy Bypass -File "tools\git\Test-PrePushHook.ps1"

pause