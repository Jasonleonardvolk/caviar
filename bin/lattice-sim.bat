@echo off
REM Dark Soliton Lattice Simulator CLI Wrapper (Windows)

setlocal enabledelayedexpansion

REM Get the directory of this script
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Set Python path
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

REM Default config file
if "%1"=="" (
    set CONFIG_FILE=conf\lattice_config.yaml
) else (
    set CONFIG_FILE=%1
)

REM Check if config exists
if not exist "%CONFIG_FILE%" (
    echo Error: Config file not found: %CONFIG_FILE%
    exit /b 1
)

REM Run simulator
echo Starting dark soliton simulation...
echo Config: %CONFIG_FILE%

python -c "import sys; import os; sys.path.insert(0, r'%PROJECT_ROOT%'); from tools.simulate_darknet import run; import logging; logging.basicConfig(level=logging.INFO, format='%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s'); sim = run(r'%CONFIG_FILE%'); print('Running simulation...'); metrics = sim.step(1000); print(f'Phase drift: {metrics[\"phase_drift\"]:.6f} rad'); print(f'Energy: {metrics[\"energy\"]:.3f}'); print(f'Steps completed: {metrics[\"step_count\"]}'); print('Simulation done ✓') if metrics['phase_drift'] < 0.02 else sys.exit(1)"

set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==0 (
    echo Simulation completed successfully
) else (
    echo Simulation failed with exit code: %EXIT_CODE%
)

exit /b %EXIT_CODE%
