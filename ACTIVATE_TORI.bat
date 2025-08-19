@echo off
echo ========================================
echo Activating TORI Virtual Environment
echo ========================================
echo.

REM Always start from project root
cd /d C:\Users\jason\Desktop\tori\kha

REM Activate venv
call .venv\Scripts\activate

REM Verify activation
echo Activated Python:
where python
python -c "import sys; print('Using:', sys.executable)"

echo.
echo You can now use 'python' commands safely!
echo.
echo Quick tests:
echo - python -c "import penrose_engine_rs; print('Penrose OK')"
echo - python enhanced_launcher.py
echo.
