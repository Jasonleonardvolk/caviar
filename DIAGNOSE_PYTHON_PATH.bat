@echo off
echo ========================================
echo Python Environment Diagnostic
echo ========================================
echo.

echo [1] Which Python executables are on PATH:
where python
echo.

echo [2] Current Python details:
python -c "import sys, os; print('sys.executable:', sys.executable); print('VIRTUAL_ENV:', os.getenv('VIRTUAL_ENV', 'Not set'))"
echo.

echo [3] Activating venv and checking again:
call .venv\Scripts\activate
echo.
echo After activation:
where python
python -c "import sys, os; print('sys.executable:', sys.executable); print('VIRTUAL_ENV:', os.getenv('VIRTUAL_ENV', 'Not set'))"
echo.

echo [4] Testing import with venv Python explicitly:
.venv\Scripts\python -c "import penrose_engine_rs, sys; print('SUCCESS: Rust backend ready from', sys.executable)"
echo.

pause
