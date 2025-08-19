@echo off
echo Running Neural Lyapunov Training & Certification Demo
echo ====================================================

REM Make sure we're in the right directory
cd %~dp0

REM Install required packages if not already installed
pip install numpy torch matplotlib

REM Set Python path to include parent directory
set PYTHONPATH=%PYTHONPATH%;..\..\

REM Create outputs directory if it doesn't exist
if not exist outputs mkdir outputs

REM Run the Van der Pol neural Lyapunov demo
python -m alan_backend.elfin.stability.demos.vdp_neural_lyap_demo ^
    --initial-steps 1000 ^
    --cert-rounds 3 ^
    --fine-tune-steps 200 ^
    --domain-size 2.5 ^
    --output-dir outputs

REM Check if the demo ran successfully
if %ERRORLEVEL% NEQ 0 (
    echo Error: Demo failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Demo completed successfully!
echo See outputs directory for saved model and plots.

REM Pause to see the output
pause
