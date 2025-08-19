@echo off
echo ================================================================================
echo QUICK OMEGA TEST - PROOF OF CONCEPT
echo Testing exotic topologies with smaller matrices
echo ================================================================================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo Running quick omega test...
echo (This should take less than 1 minute)
echo.

python quick_omega_test.py

echo.
echo ================================================================================
echo Quick test complete!
echo Check quick_omega_test.png for results
echo.
echo To run the full benchmark, use: RUN_OMEGA_BENCHMARK.bat
echo ================================================================================

pause
