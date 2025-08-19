@echo off
echo Setting ScholarSphere environment variables...

REM Replace with your actual API key
set SCHOLARSPHERE_API_KEY=your-scholarsphere-api-key-here
set SCHOLARSPHERE_API_URL=https://api.scholarsphere.org
set SCHOLARSPHERE_BUCKET=concept-diffs

REM Enable entropy pruning
set TORI_ENABLE_ENTROPY_PRUNING=1
set TORI_DISABLE_ENTROPY_PRUNE=

echo.
echo Environment variables set:
echo   SCHOLARSPHERE_API_KEY = %SCHOLARSPHERE_API_KEY%
echo   SCHOLARSPHERE_API_URL = %SCHOLARSPHERE_API_URL%
echo   SCHOLARSPHERE_BUCKET = %SCHOLARSPHERE_BUCKET%
echo   TORI_ENABLE_ENTROPY_PRUNING = %TORI_ENABLE_ENTROPY_PRUNING%
echo.
echo IMPORTANT: Replace 'your-scholarsphere-api-key-here' with your actual API key!
echo.
echo Now run: poetry run python enhanced_launcher.py
pause
