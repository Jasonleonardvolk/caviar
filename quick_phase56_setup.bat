@echo off
echo ========================================
echo Phase 5-6 Quick Setup
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing deepdiff...
pip install deepdiff>=6.7

echo.
echo Running Penrose performance tests...
pytest tests\test_penrose.py -v

echo.
echo Ready to test Phase 6!
echo.
echo Next: Restart TORI and run test_diff_endpoint.bat
pause
