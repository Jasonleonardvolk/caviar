@echo off
echo ========================================
echo Installing Phase 6 Dependencies
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing deepdiff for structural diffs...
pip install deepdiff>=6.7

echo.
echo Dependencies installed!
echo.
echo Next steps:
echo 1. Restart TORI: python enhanced_launcher.py --no-browser
echo 2. Test the endpoint: .\test_diff_endpoint.bat
echo.
pause
