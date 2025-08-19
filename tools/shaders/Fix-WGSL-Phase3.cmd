@echo off
REM Fix-WGSL-Phase3.cmd
REM Alignment and reserved-identifier fixes. Keeps window open.

set SCRIPT_DIR=%~dp0
echo Launching Phase-3 WGSL fixer from %SCRIPT_DIR%
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Fix-WGSL-Phase3.ps1" -Apply
echo.
echo Done. Reports (if any) are in: %SCRIPT_DIR%reports\
echo Backups are in: ..\..\frontend\shaders.bak\phase3\ <timestamp>\
pause
