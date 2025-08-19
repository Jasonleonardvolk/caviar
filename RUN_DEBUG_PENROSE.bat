@echo off
echo ================================================================================
echo DEBUG: Why isn't Penrose running?
echo ================================================================================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo Testing with --experimental flag...
set TORI_ENABLE_EXOTIC=1
python debug_penrose_inclusion.py --experimental

echo.
echo ================================================================================
echo Now let's add explicit debug to robust_omega_test.py
echo ================================================================================
echo.

REM Add a debug line to the test
python -c "print('Checking if penrose is in the loop...')"
findstr /n "penrose" robust_omega_test.py

echo.
pause
