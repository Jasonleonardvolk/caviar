@echo off
echo 🚀 STARTING TORI PRODUCTION SERVER
echo =====================================
echo.
echo 📊 This starts the complete TORI system:
echo    - PDF Extraction Service
echo    - Soliton Memory System
echo    - WebSocket Progress Tracking
echo    - All API Endpoints
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo 🌊 Starting TORI with Soliton Memory...
echo.

REM Start the complete API server (includes everything)
python start_dynamic_api.py

echo.
echo 🛑 TORI Production server stopped.
pause
