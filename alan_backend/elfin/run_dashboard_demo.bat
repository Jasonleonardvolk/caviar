@echo off
echo Running ELFIN Stability Dashboard Demo
echo =====================================

REM Make sure we're in the right directory
cd %~dp0

REM Install required packages if not already installed
pip install flask numpy

REM Set Python path to include parent directory
set PYTHONPATH=%PYTHONPATH%;..\..\

REM Run the dashboard demo
python examples/dashboard_demo.py

REM Check if the demo ran successfully
if %ERRORLEVEL% NEQ 0 (
    echo Error: Demo failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Demo completed successfully!

REM Pause to see the output
pause
