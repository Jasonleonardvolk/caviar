@echo off
echo =======================================================
echo 🧠 TORI with Real Cognitive Engine - Full System Launch
echo =======================================================
echo This will start BOTH systems:
echo 1. Node.js Cognitive Engine Bridge (port 4321)
echo 2. Python FastAPI Backend with chat integration
echo =======================================================

cd /d "%~dp0"

echo.
echo 📦 Installing Node.js dependencies...
call npm install
if errorlevel 1 (
    echo ❌ Failed to install Node.js dependencies
    pause
    exit /b 1
)

echo.
echo 🧠 Starting Cognitive Engine Bridge on port 4321...
start "Cognitive Bridge" cmd /k "node cognitive-bridge.js"

echo.
echo ⏳ Waiting 5 seconds for cognitive bridge to start...
timeout /t 5 /nobreak > nul

echo.
echo 🧠 Testing cognitive bridge health...
curl -s http://localhost:4321/health > nul
if errorlevel 1 (
    echo ⚠️ Cognitive bridge may not be ready yet, but continuing...
) else (
    echo ✅ Cognitive bridge is responding
)

echo.
echo 🚀 Starting Python FastAPI backend with cognitive integration...
echo 📍 The backend will now use your REAL cognitive engine!
echo.

python start_unified_tori.py

echo.
echo 🔄 System shutdown complete
pause
