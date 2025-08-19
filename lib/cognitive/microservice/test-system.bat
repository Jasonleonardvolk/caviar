@echo off
echo 🧪 Running TORI Cognitive System Integration Tests...
echo ===================================================

cd /d "C:\Users\jason\Desktop\tori\kha\lib\cognitive\microservice"

REM Check if both services are running
echo 🔍 Checking if services are running...

REM Test Node.js microservice
curl -s http://localhost:4321/api/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js Cognitive Microservice not running on port 4321
    echo Please run: start-cognitive-microservice.bat
    echo.
    pause
    exit /b 1
)

REM Test FastAPI bridge  
curl -s http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    echo ❌ FastAPI Bridge not running on port 8000
    echo Please run: start-fastapi-bridge.bat
    echo.
    pause
    exit /b 1
)

echo ✅ Both services are running!
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run the comprehensive test suite
echo 🚀 Running comprehensive integration tests...
python test_integration.py

echo.
echo 📋 Test completed! Check the results above.
echo 📊 Detailed results saved to test_results_*.json
echo.
pause
