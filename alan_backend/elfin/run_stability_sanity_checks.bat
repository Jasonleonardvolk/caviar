@echo off
echo ELFIN Neural Lyapunov Framework - Post-Merge Sanity Checks
echo =========================================================

REM Make sure we're in the right directory
cd %~dp0

REM Install required packages if not already installed
pip install pytest pytest-cov coverage

REM Set Python path to include parent directory
set PYTHONPATH=%PYTHONPATH%;%~dp0..\..

REM Ensure outputs directory exists
if not exist outputs mkdir outputs

echo.
echo Running all sanity checks - This might take several minutes...
echo.

REM Run all checks - Run the script directly to avoid importing the whole package
python stability\tests\run_sanity_checks.py --check all

REM Check if the tests ran successfully
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Sanity checks failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo You can also run individual checks with:
echo   run_stability_sanity_checks.bat roundtrip    - Trainer/Verifier round-trip
echo   run_stability_sanity_checks.bat pd           - True positive definiteness
echo   run_stability_sanity_checks.bat bigm         - Big-M tightness
echo   run_stability_sanity_checks.bat fallback     - Solver fall-back
echo   run_stability_sanity_checks.bat coverage     - Unit test coverage

echo.
echo Sanity checks completed!

REM Pause to see the output
pause

REM Handle specific check request if provided as parameter
if "%1"=="" goto end

echo.
echo Running specific check: %1
echo.

python stability\tests\run_sanity_checks.py --check %1

:end
