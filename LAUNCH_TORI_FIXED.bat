@echo off
echo Starting TORI System...

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found!
    exit /b 1
)

REM Start backend
echo Starting backend API...
start /B python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM Wait for backend to start
timeout /t 5 /nobreak > nul

REM Start frontend
echo Starting frontend...
cd tori_ui_svelte
start cmd /k "npm run dev -- --host"
cd ..

echo.
echo TORI System started!
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:5173
echo.
echo Press any key to stop all services...
pause > nul

REM Kill processes
taskkill /F /IM node.exe 2>nul
taskkill /F /IM python.exe 2>nul
