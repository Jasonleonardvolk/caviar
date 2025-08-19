@echo off
echo 🐍 Starting TORI Cognitive FastAPI Bridge...
echo ============================================

cd /d "C:\Users\jason\Desktop\tori\kha\lib\cognitive\microservice"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python first.
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist ".venv" (
    echo 🔧 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install requirements
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Start the FastAPI bridge
echo 🚀 Starting FastAPI bridge on port 8000...
echo 📡 API endpoints will be available at http://localhost:8000/api/
echo ⭐ Main chat: POST http://localhost:8000/api/chat
echo 🧠 Smart ask: POST http://localhost:8000/api/smart/ask
echo 📊 Status: GET http://localhost:8000/api/status
echo 📚 Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the service
echo ============================================

python cognitive_bridge.py

pause
