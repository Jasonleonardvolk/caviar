@echo off
echo 🔍 TORI API CRASH CAPTURE
echo ========================
echo.
echo This will run the API server with comprehensive crash detection
echo All output will be logged to files for analysis
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo 🚀 Starting crash capture...
python direct_crash_capture.py

echo.
echo 🔍 Crash capture complete
echo 📝 Check the logs/ directory for detailed crash information
echo.
pause
