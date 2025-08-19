@echo off
echo ================================================================================
echo PENROSE MICROKERNEL DIAGNOSTICS
echo ================================================================================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

set TORI_ENABLE_EXOTIC=1
python diagnose_penrose_microkernel.py

echo.
pause
