@echo off
echo ============================================================
echo IRIS Complete E2E Verification (Fixed Version)
echo ============================================================
echo.
echo This script runs the complete verification without prompts
echo and generates detailed reports including diagnostics.
echo.
echo Starting verification...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0Verify-EndToEnd-Fixed.ps1" -QuickBuild

echo.
if %ERRORLEVEL% EQU 0 (
    echo ============================================================
    echo BUILD PASSED - GO FOR RELEASE!
    echo ============================================================
    echo Check the reports folder for detailed results.
) else (
    echo ============================================================
    echo BUILD FAILED - NO-GO
    echo ============================================================
    echo.
    echo Running diagnostic analyzer...
    echo.
    powershell -ExecutionPolicy Bypass -File "%~dp0Analyze-BuildFailure.ps1"
    echo.
    echo See the reports folder for detailed failure analysis.
)

echo.
pause
