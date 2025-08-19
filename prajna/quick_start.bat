@echo off
REM Prajna Quick Start Script for Windows
REM =====================================
REM 
REM This script sets up and launches Prajna with minimal configuration.
REM Perfect for getting started quickly or testing the system.

echo 🧠 Prajna Quick Start - TORI's Voice and Language Model
echo =======================================================

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install --quiet --upgrade pip
pip install --quiet fastapi uvicorn websockets aiohttp numpy scikit-learn

REM Optional dependencies
echo 🔧 Installing optional dependencies...
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>nul || echo ⚠️  PyTorch not installed (using fallback)
pip install --quiet transformers 2>nul || echo ⚠️  Transformers not installed (using fallback)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "snapshots" mkdir snapshots

REM Run tests
echo 🧪 Running Prajna tests...
python test_prajna.py
if errorlevel 1 (
    echo ❌ Tests failed! Please check the output above.
    pause
    exit /b 1
)

echo.
echo 🎉 All tests passed! Starting Prajna in demo mode...
echo.
echo 🔗 API will be available at: http://localhost:8001
echo 🔗 Health check: http://localhost:8001/api/health
echo 🔗 API docs: http://localhost:8001/docs
echo.
echo 📝 To test Prajna:
echo    curl -X POST http://localhost:8001/api/answer ^
echo      -H "Content-Type: application/json" ^
echo      -d "{\"user_query\": \"What is Prajna?\"}"
echo.
echo 🛑 Press Ctrl+C to stop Prajna
echo.

REM Start Prajna
python start_prajna.py --demo --log-level INFO

pause
