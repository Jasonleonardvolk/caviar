@echo off
echo ========================================
echo Installing TORI Complete Dependencies
echo ========================================
echo.

REM Make sure we're in venv
call .venv\Scripts\activate

echo Installing core dependencies...
pip install "networkx>=3.0"
pip install "pydantic>=2.6" "pydantic-settings>=2.0"
pip install scipy
pip install requests psutil uvicorn

echo.
echo Installing other common dependencies that might be needed...
pip install numpy pandas scikit-learn matplotlib

echo.
echo Freezing dependencies...
pip freeze > requirements-dev.lock

echo.
echo ========================================
echo Testing TORI with Penrose...
echo ========================================
python enhanced_launcher.py --require-penrose --no-browser

pause
