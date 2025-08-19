@echo off
REM Fix-WGSL-Apply.cmd
REM Run the fixer with common flags and keep the window open.

set SCRIPT_DIR=%~dp0
echo Launching Fix-WGSL.ps1 from %SCRIPT_DIR%
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Fix-WGSL.ps1" -Apply -AutoClampWorkgroup
echo.
echo Done. Reports (if any) are in: %SCRIPT_DIR%reports\
echo Backups are in: ..\..\frontend\shaders.bak\auto_fixes\ <timestamp>\
pause
