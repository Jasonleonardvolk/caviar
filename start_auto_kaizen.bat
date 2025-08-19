@echo off
REM 🤖 TORI Auto-Kaizen Monitor Launcher
REM Starts TORI's self-improvement consciousness

echo =====================================
echo 🧠 TORI AUTO-KAIZEN MONITOR
echo =====================================
echo.
echo Starting TORI's self-awareness system...
echo This will monitor performance and create improvement tickets automatically.
echo.
echo Press Ctrl+C to stop monitoring
echo =====================================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start the monitor
python registry\kaizen\auto_kaizen.py monitor

pause
