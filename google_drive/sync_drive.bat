@echo off
echo ========================================
echo Google Drive Sync for Tori Project
echo ========================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha

echo Installing required dependencies...
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

echo.
echo Starting Google Drive synchronization...
python google_drive\drive_sync.py

echo.
echo ========================================
echo Sync completed!
echo ========================================
pause
