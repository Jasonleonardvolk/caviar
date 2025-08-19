@echo off
echo Installing TORI dependencies...
echo.

REM Make sure we're in venv
call .venv\Scripts\activate

REM Install all requirements
pip install -r requirements.txt
pip install -r requirements-dev.txt

REM Install additional launcher requirements
pip install requests psutil uvicorn

echo.
echo Dependencies installed! Testing again...
echo.

REM Test the launcher again
python enhanced_launcher.py --require-penrose --no-browser

pause
