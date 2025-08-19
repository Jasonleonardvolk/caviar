@echo off
echo ============================================
echo Checking Release Verification Setup
echo ============================================
echo.

cd /d D:\Dev\kha\tools\release

echo Checking for script files:
echo.

if exist "Verify-EndToEnd-Improved.ps1" (
    echo [OK] Verify-EndToEnd-Improved.ps1 exists
) else (
    echo [MISSING] Verify-EndToEnd-Improved.ps1
)

if exist "IrisOneButton_NonInteractive.ps1" (
    echo [OK] IrisOneButton_NonInteractive.ps1 exists - NO PROMPTS
    set NOPROMPT=true
) else (
    echo [MISSING] IrisOneButton_NonInteractive.ps1
    set NOPROMPT=false
)

if exist "IrisOneButton.ps1" (
    echo [OK] IrisOneButton.ps1 exists (original)
) else (
    echo [MISSING] IrisOneButton.ps1
)

echo.
echo ============================================
echo RESULT:
echo ============================================
echo.

if "%NOPROMPT%"=="true" (
    echo STATUS: READY FOR AUTOMATION
    echo The improved script will use IrisOneButton_NonInteractive.ps1
    echo You will NOT see any prompts during verification.
    echo.
    echo Run this command:
    echo powershell -ExecutionPolicy Bypass -File .\Verify-EndToEnd-Improved.ps1 -QuickBuild
) else (
    echo STATUS: WILL HAVE PROMPTS
    echo The improved script will fall back to IrisOneButton.ps1
    echo You WILL see "Open release folder?" prompt.
    echo.
    echo To fix, run:
    echo powershell -ExecutionPolicy Bypass -File .\Verify-EndToEnd-Improved.ps1 -QuickBuild
    echo Then press N when prompted.
)

echo.
pause
