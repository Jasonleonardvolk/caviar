@echo off
echo ================================================================================
echo DEBUGGING ACCURACY ISSUE
echo ================================================================================
echo.
echo Setting thread control...
set NUMBA_THREADING_LAYER=omp
set NUMBA_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1

cd /d "C:\Users\jason\Desktop\tori\kha"

python tests\test_debug_accuracy.py

echo.
pause
