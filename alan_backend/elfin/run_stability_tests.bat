@echo off
echo Running ELFIN Stability Framework Tests
echo ======================================

REM Make sure we're in the right directory
cd %~dp0

REM Install required packages if not already installed
pip install pytest numpy torch

REM Set Python path to include parent directory
set PYTHONPATH=%PYTHONPATH%;..\..\

REM Run all stability tests with verbose output
pytest -xvs alan_backend/elfin/stability/tests/

REM Check if the tests ran successfully
if %ERRORLEVEL% NEQ 0 (
    echo Error: Tests failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo All tests passed successfully!

REM Pause to see the output
pause
