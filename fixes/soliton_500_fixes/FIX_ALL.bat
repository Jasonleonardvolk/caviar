@echo off
echo ========================================
echo    SOLITON API 500 ERROR FIX
echo ========================================
echo.

echo Step 1: Testing current status...
python test_soliton_api.py
echo.

pause
echo.
echo Step 2: Applying all fixes...
python fix_soliton_main.py
echo.

pause
echo.
echo Step 3: Applying fixed API route...
python apply_soliton_api_fix.py
echo.

echo ========================================
echo    FIX COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Run start_backend_debug.py to start the API
echo 2. Run test_soliton_api.py again to verify fixes
echo.
pause
