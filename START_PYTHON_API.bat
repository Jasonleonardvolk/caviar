@echo off
echo 🚀 STARTING TORI PYTHON EXTRACTION API
echo =======================================
echo.
echo 🎯 This will start the Python concept extraction API on port 8002
echo 📍 API will be available at: http://localhost:8002
echo 🔗 Health check: http://localhost:8002/health
echo 📄 API docs: http://localhost:8002/docs
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo.
echo 🔍 Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo 📦 Installing FastAPI dependencies...
pip install fastapi uvicorn yake keybert sentence-transformers spacy
if %errorlevel% neq 0 (
    echo ⚠️ Some dependencies may already be installed - continuing...
)

echo.
echo 🧠 Checking spaCy model...
python -c "import spacy; spacy.load('en_core_web_lg')" 2>nul
if %errorlevel% neq 0 (
    echo 📥 Downloading spaCy model...
    python -m spacy download en_core_web_lg
)

echo.
echo 🚀 Starting Python Extraction API Server...
echo 📍 Server will be available at: http://localhost:8002
echo 🔗 Health check: http://localhost:8002/health
echo 📄 API documentation: http://localhost:8002/docs
echo.
echo ⏱️ STARTING API SERVER NOW...
echo.

python python_extraction_api.py

echo.
echo ❌ API server stopped. Press any key to restart...
pause
goto start
