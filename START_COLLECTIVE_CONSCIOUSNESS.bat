@echo off
REM TORI Collective Consciousness Startup
REM Launches multiple TORI agents with braid fusion and introspection loops

echo ================================================================================
echo                 TORI COLLECTIVE CONSCIOUSNESS SYSTEM v5.0
echo          "Individual thoughts become collective wisdom through connection"
echo ================================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check for async support
python -c "import asyncio" 2>nul
if errorlevel 1 (
    echo ERROR: asyncio not available. Please upgrade Python.
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "multi_agent\local_exchange" mkdir "multi_agent\local_exchange"
if not exist "introspection_logs" mkdir "introspection_logs"

echo.
echo Configuration:
echo   - Number of agents: 3
echo   - Sync mode: WORMHOLE (direct concept transfer)
echo   - Introspection interval: 5 minutes
echo   - Collective introspection: 10 minutes
echo.

echo Starting collective consciousness...
echo.

REM Run the collective consciousness demo
python multi_agent_metacognition.py

if errorlevel 1 (
    echo.
    echo ERROR: Collective consciousness failed to initialize
    echo Check error messages above
) else (
    echo.
    echo Collective consciousness terminated gracefully
)

echo.
echo ================================================================================
echo Introspection logs saved to: introspection_logs\
echo Collective insights saved to: multi_agent\collective_insights.json
echo.
echo "Through many minds, one understanding emerges."
echo ================================================================================
echo.
pause
