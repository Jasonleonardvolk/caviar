@echo off
REM Launch script for TORI Chaos-Enhanced System (Windows)

echo 🚀 Launching TORI Chaos-Enhanced System...
echo ==================================

REM Set environment variables
set PYTHONPATH=%CD%;%PYTHONPATH%
set CHAOS_EXPERIMENT=1

REM Check Python version
python --version

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "snapshots" mkdir snapshots
if not exist "conf" mkdir conf

REM Check if config files exist
if not exist "conf\lattice_config.yaml" (
    echo Warning: lattice_config.yaml not found
)

REM Launch options
if "%1"=="" goto full
if "%1"=="full" goto full
if "%1"=="test" goto test
if "%1"=="monitor" goto monitor
if "%1"=="websocket" goto websocket
if "%1"=="demo" goto demo
goto usage

:full
echo Starting full system...
python tori_master.py
goto end

:test
echo Running integration tests...
python -m pytest integration_tests.py -v
goto end

:monitor
echo Starting health monitor...
python system_health_monitor.py --simple
goto end

:websocket
echo Starting WebSocket server only...
python services\metrics_ws.py
goto end

:demo
echo Running demo...
python -c "import asyncio; from tori_master import TORIMaster; async def demo(): master = TORIMaster(); await master.start(); queries = ['What is consciousness?', 'Explore patterns in quantum systems', 'Create a novel approach to AI safety']; [print(f'\nQuery: {q}\nResponse: {(await master.process_query(q, {\"enable_chaos\": True}))[\"response\"][:200]}...') for q in queries]; await master.stop(); asyncio.run(demo())"
goto end

:usage
echo Usage: %0 [full^|test^|monitor^|websocket^|demo]
exit /b 1

:end
echo ==================================
echo TORI shutdown complete
