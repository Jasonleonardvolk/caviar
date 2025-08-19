@echo off
REM Fix-WGSL.cmd
REM Launcher for Fix-WGSL.ps1 with ExecutionPolicy Bypass

set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Fix-WGSL.ps1" %*
