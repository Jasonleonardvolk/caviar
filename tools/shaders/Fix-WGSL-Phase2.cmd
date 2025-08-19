@echo off
echo Phase-2 WGSL Mechanical Fixes
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "Fix-WGSL-Phase2.ps1" -Apply
pause
