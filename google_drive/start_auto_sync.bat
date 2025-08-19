@echo off
echo ========================================
echo Starting Google Drive Auto-Sync Service
echo ========================================
echo.
echo This service will run continuously and sync your files
echo with Google Drive at regular intervals.
echo.
echo Press Ctrl+C to stop the service
echo.
echo ========================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha

REM Check if dependencies are installed
python -c "import google.auth" 2>NUL
if errorlevel 1 (
    echo Installing required dependencies...
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
    echo.
)

REM Start the auto-sync service
python google_drive\auto_sync_service.py

echo.
echo ========================================
echo Auto-Sync Service Stopped
echo ========================================
pause
