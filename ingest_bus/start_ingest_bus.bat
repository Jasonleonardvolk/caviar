@echo off
echo Starting ingest_bus service...

:: Create data directory if it doesn't exist
if not exist data\jobs mkdir data\jobs

:: Ensure Python is available
python --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.9+ and try again.
    exit /b 1
)

:: Check if requirements are installed
echo Checking dependencies...
pip show fastapi > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

:: Start the service
echo Starting ingest_bus on http://localhost:8000
echo Metrics available on http://localhost:8081/metrics
echo Press Ctrl+C to stop the service

python -m src.main --host 0.0.0.0 --port 8000
