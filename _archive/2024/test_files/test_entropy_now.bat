@echo off
echo.
echo ========================================
echo Testing Entropy Pruning with AV Mock
echo ========================================
echo.

echo Created av.py mock module to fix compatibility issue.
echo.

echo Running test...
python test_entropy_direct.py

echo.
echo ========================================
echo If you see "ENTROPY PRUNING IS WORKING!" above,
echo then everything is fixed and ready to use!
echo ========================================
echo.
pause
