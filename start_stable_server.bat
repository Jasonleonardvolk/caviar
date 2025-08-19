@echo off
echo ================================
echo   TORI Stable FastAPI Server
echo ================================
echo.

REM Change to the directory containing this script
cd /d %~dp0

echo 🚀 Starting stable PDF ingestion server...
echo 📂 Directory: %CD%
echo 🌐 URL: http://localhost:8002
echo 🔧 File watching: DISABLED (for stability)
echo.
echo Press Ctrl+C to stop the server
echo ================================
echo.

python run_stable_server.py

echo.
echo Server stopped.
pause
