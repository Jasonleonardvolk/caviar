@echo off
echo ========================================
echo Installing Test Dependencies
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing missing test dependencies...
pip install reportlab aiofiles

echo.
echo Running Penrose performance tests...
pytest tests\test_penrose.py -v

echo.
echo Done! Dependencies installed and tests run.
pause
