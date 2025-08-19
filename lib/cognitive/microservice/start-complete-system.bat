@echo off
echo 🔥 Starting Complete TORI Cognitive System...
echo =============================================
echo.
echo This will start both:
echo 1. Node.js Cognitive Microservice (port 4321)
echo 2. FastAPI Python Bridge (port 8000)
echo.

REM Start Node.js microservice in background
echo 🧠 Starting Node.js Cognitive Microservice...
start "TORI Cognitive Microservice" cmd /k "start-cognitive-microservice.bat"

REM Wait a moment for the microservice to start
timeout /t 3 /nobreak >nul

REM Start FastAPI bridge
echo 🐍 Starting FastAPI Bridge...
start "TORI FastAPI Bridge" cmd /k "start-fastapi-bridge.bat"

echo.
echo ✅ TORI Cognitive System is starting up!
echo.
echo 📡 Available endpoints:
echo   Node.js API: http://localhost:4321/api/
echo   FastAPI API: http://localhost:8000/api/
echo   Documentation: http://localhost:8000/docs
echo.
echo ⭐ Your main endpoint for Python/FastAPI integration:
echo   POST http://localhost:8000/api/chat
echo.
echo 🎯 Test with curl:
echo   curl -X POST "http://localhost:8000/api/chat" \
echo        -H "Content-Type: application/json" \
echo        -d "{\"message\":\"Hello cognitive engine\", \"glyphs\":[\"anchor\",\"concept-synthesizer\",\"return\"]}"
echo.
echo Press any key to close this window (services will continue running)
pause >nul
