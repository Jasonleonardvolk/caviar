@echo off
echo.
echo ========================================
echo Fixing AV/Torchvision Compatibility
echo ========================================
echo.

echo Option 1: Try compatible av version...
pip uninstall -y av
pip install av==10.0.0

echo.
echo Testing...
python -c "import av; print(f'av {av.__version__} installed')"
python -c "import sentence_transformers; print('sentence_transformers OK')"

if errorlevel 1 (
    echo.
    echo Option 2: Installing different torchvision...
    pip uninstall -y torchvision
    pip install torchvision==0.16.0
    
    echo.
    echo Testing again...
    python -c "import sentence_transformers; print('sentence_transformers OK')"
)

if errorlevel 1 (
    echo.
    echo Option 3: Creating workaround...
    python create_av_workaround.py
)

echo.
echo ========================================
echo Final test...
echo ========================================
python test_entropy_state.py

pause
