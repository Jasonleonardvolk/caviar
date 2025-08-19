@echo off
REM Apply Non-Interactive Fix for Release Verification

echo ============================================
echo Applying Non-Interactive Fix
echo ============================================
echo.

REM Backup original IrisOneButton.ps1
echo Backing up original IrisOneButton.ps1...
if exist "IrisOneButton.ps1" (
    copy /Y "IrisOneButton.ps1" "IrisOneButton.ps1.original"
    echo Backup created: IrisOneButton.ps1.original
) else (
    echo WARNING: IrisOneButton.ps1 not found
)

REM Copy the non-interactive version
echo.
echo Applying non-interactive version...
if exist "IrisOneButton_NonInteractive.ps1" (
    copy /Y "IrisOneButton_NonInteractive.ps1" "IrisOneButton.ps1"
    echo SUCCESS: IrisOneButton.ps1 updated with NonInteractive support
) else (
    echo ERROR: IrisOneButton_NonInteractive.ps1 not found
    echo Please ensure you have the fixed version
    exit /b 1
)

echo.
echo ============================================
echo Fix Applied Successfully!
echo ============================================
echo.
echo You can now run verification without prompts:
echo powershell -ExecutionPolicy Bypass -File .\Verify-EndToEnd-Improved.ps1 -QuickBuild
echo.
echo To restore original (if needed):
echo copy /Y IrisOneButton.ps1.original IrisOneButton.ps1
echo.
pause
