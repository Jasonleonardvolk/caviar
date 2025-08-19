@echo off
REM Start Cognitive Interface Service
REM =================================

echo Starting Cognitive Interface Service...
echo.

REM Navigate to project root
cd /d C:\Users\jason\Desktop\tori\kha

REM Set PYTHONPATH
set PYTHONPATH=%cd%
echo PYTHONPATH: %PYTHONPATH%
echo.

REM Kill any existing process on port 5173
echo Checking for existing processes on port 5173...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    echo Killing process %%a
    taskkill /PID %%a /F 2>nul
)

REM Start the service
echo Starting uvicorn server...
echo.
echo Service will be available at: http://localhost:5173/docs
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn ingest_pdf.cognitive_interface:app --host 0.0.0.0 --port 5173 --reload

pause
