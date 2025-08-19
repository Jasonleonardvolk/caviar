@echo off
echo Downloading depth estimation model...
echo.

set MODEL_PATH=C:\Users\jason\Desktop\tori\kha\standalone-holo\public\models\depth_estimator.onnx
set MIDAS_URL=https://huggingface.co/julienkay/sentis-MiDaS/resolve/b867253b86ef4cef1cfda70e8fbcf72fb27eaa3e/midas_v21_small_256.onnx?download=true

echo Downloading MiDaS v2.1 Small (66MB)...
curl -L "%MIDAS_URL%" -o "%MODEL_PATH%"

if exist "%MODEL_PATH%" (
    echo.
    echo SUCCESS: Model downloaded to:
    echo %MODEL_PATH%
) else (
    echo.
    echo FAILED: Could not download model.
    echo Please download manually from:
    echo %MIDAS_URL%
    echo And save to:
    echo %MODEL_PATH%
)

echo.
echo Note: WaveOp model is optional - app will use fallback if not present
pause