@echo off
REM Run TORI Compaction Now
REM Simple wrapper to run compaction manually

cd /d "%~dp0"
python compact_all_meshes.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Compaction failed with error code %ERRORLEVEL%
    pause
)
