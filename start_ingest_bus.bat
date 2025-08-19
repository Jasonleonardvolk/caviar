@echo off
echo =====================================
echo   TORI Ingest Bus Service Launcher  
echo =====================================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha\ingest_bus"

echo [1/3] Checking dependencies...
if not exist requirements.txt (
    echo ❌ requirements.txt not found!
    pause
    exit /b 1
)

echo [2/3] Installing/updating dependencies...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Warning: Some dependencies may have issues, proceeding anyway...
)

echo [3/3] Starting Ingest Bus service...
echo.
echo 🚀 Starting FastAPI service on port 8080...
echo    - API Documentation: http://localhost:8080/docs
echo    - Health Check: http://localhost:8080/health
echo    - Integration: Connected to TORI MCP ecosystem
echo.

python main.py
if errorlevel 1 (
    echo.
    echo ❌ Failed to start Ingest Bus service
    echo    Check logs for details
    pause
    exit /b 1
)

echo.
echo ✅ Ingest Bus service stopped
pause
