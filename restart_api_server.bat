@echo off
echo Restarting TORI API server...
echo --------------------------------------------------

echo Stopping existing servers...
taskkill /F /IM uvicorn.exe 2>nul
for /f "tokens=2" %%i in ('netstat -ano ^| findstr :8002') do taskkill /F /PID %%i 2>nul

timeout /t 2 /nobreak >nul

echo.
echo Starting API server on port 8002...
echo Press Ctrl+C to stop
echo --------------------------------------------------

poetry run uvicorn prajna_api:app --host 0.0.0.0 --port 8002 --reload --log-level info
