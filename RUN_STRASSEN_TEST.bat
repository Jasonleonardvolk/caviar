@echo off
echo ================================================================================
echo STRASSEN-OPTIMIZED SOLITON PHYSICS TEST
echo Testing O(n^2.807) complexity with Strassen recursion
echo ================================================================================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

python tests\test_strassen_soliton.py

echo.
pause
