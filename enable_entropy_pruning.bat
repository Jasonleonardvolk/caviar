@echo off
echo Setting TORI environment variables...
set TORI_ENABLE_ENTROPY_PRUNING=1
set TORI_DISABLE_ENTROPY_PRUNE=
echo Entropy pruning enabled
echo.
echo Now run: poetry run python enhanced_launcher.py
pause
