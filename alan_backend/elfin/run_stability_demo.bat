@echo off
echo Running ELFIN Stability Framework Demo
echo =====================================

REM Make sure we're in the right directory
cd %~dp0

REM Set Python path to include parent directory
set PYTHONPATH=%PYTHONPATH%;..\..\

REM Run the demo
python examples/elfin_stability_demo.py

REM Check if the demo ran successfully
if %ERRORLEVEL% NEQ 0 (
    echo Error: Demo failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Demo completed successfully!
echo Results saved in elfin_stability_demo.png

REM Pause to see the output
pause
