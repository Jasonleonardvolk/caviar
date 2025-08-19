@echo off
echo 🧠 Starting TORI Cognitive Engine Microservice...
echo ================================================

cd /d "C:\Users\jason\Desktop\tori\kha\lib\cognitive\microservice"

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js first.
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "node_modules" (
    echo 📦 Installing Node.js dependencies...
    npm install
)

REM Start the TypeScript microservice
echo 🚀 Starting cognitive microservice on port 4321...
echo 📡 Endpoints will be available at http://localhost:4321/api/
echo ⭐ Main processing: POST http://localhost:4321/api/engine
echo 📊 Status: GET http://localhost:4321/api/status
echo ❤️ Health: GET http://localhost:4321/api/health
echo.
echo Press Ctrl+C to stop the service
echo ================================================

npx ts-node cognitive-microservice.ts

pause
