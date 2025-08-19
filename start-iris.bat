@echo off
title IRIS Quick Start (Isolated Environments)

echo === IRIS QUICK START (ISOLATED) ===
echo.

echo Creating/Using isolated Penrose environment...
cd services\penrose

if not exist ".venv" (
    echo Creating new isolated venv for Penrose...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        python -m venv .venv
    )
    echo Installing Penrose dependencies in isolation...
    .venv\Scripts\pip install -r requirements.txt
)

echo Starting Penrose service with isolated environment...
start /B cmd /c ".venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 7401"
timeout /t 5 > nul

cd ..\..\tori_ui_svelte

echo Starting frontend development server...
call pnpm install
call pnpm dev

echo.
echo Services started!
echo.
echo Frontend: http://localhost:5173
echo Penrose:  http://localhost:7401 (isolated env)
echo.
pause