@echo off
echo Restarting TORI using enhanced_launcher.py...
echo --------------------------------------------------

echo Stopping existing TORI processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq TORI*" 2>nul
taskkill /F /IM node.exe /FI "WINDOWTITLE eq TORI*" 2>nul
for /f "tokens=2" %%i in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %%i 2>nul
for /f "tokens=2" %%i in ('netstat -ano ^| findstr :8001') do taskkill /F /PID %%i 2>nul
for /f "tokens=2" %%i in ('netstat -ano ^| findstr :8002') do taskkill /F /PID %%i 2>nul
for /f "tokens=2" %%i in ('netstat -ano ^| findstr :5173') do taskkill /F /PID %%i 2>nul

timeout /t 2 /nobreak >nul

echo.
echo Starting TORI with enhanced_launcher.py...
echo Press Ctrl+C to stop
echo --------------------------------------------------

poetry run python enhanced_launcher.py
