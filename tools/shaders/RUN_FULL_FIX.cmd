@echo off
echo ============================================
echo FULL SHADER FIX AND VALIDATION PIPELINE
echo ============================================
echo.

cd /d "%~dp0"

echo Step 1: Running Phase-2 Mechanical Fixes...
powershell -ExecutionPolicy Bypass -File "Fix-WGSL-Phase2.ps1" -Apply
echo.

echo Step 2: Running Phase-3 Alignment Fixes...
powershell -ExecutionPolicy Bypass -File "Fix-WGSL-Phase3.ps1" -Apply
echo.

echo Step 3: Running Validation...
node shader_validation_wrapper.mjs
echo.

echo ============================================
echo COMPLETE - Check reports\shader_validation_latest.json
echo ============================================
pause
