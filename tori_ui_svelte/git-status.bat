@echo off
REM Quick status check for iRis

cd /d D:\Dev\kha\tori_ui_svelte

echo =====================================
echo        iRis Project Status
echo =====================================
echo.

powershell -ExecutionPolicy Bypass -Command "& { .\tools\git\Git-Workflow.ps1 status }"

echo.
pause