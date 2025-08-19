@echo off
echo ================================================================================
echo TESTING ALL QUICK-WIN OPTIMIZATIONS
echo ================================================================================
echo.
echo Setting BLAS thread control...
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1

cd /d "C:\Users\jason\Desktop\tori\kha"

python tests\test_all_optimizations.py

echo.
pause
