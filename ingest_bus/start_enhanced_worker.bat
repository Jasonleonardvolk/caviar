@echo off
:: Ingest Bus Enhanced Worker Starter
:: This script starts the enhanced worker with math awareness

echo Starting enhanced ingest worker...

:: Set environment variables
set INGEST_BUS_URL=http://localhost:8000
set WORKER_ID=enhanced-worker-%RANDOM%
set MAX_CHUNK_SIZE=1800
set MAX_WORKERS=2
set PYTHONPATH=%PYTHONPATH%;%~dp0

:: Check if MathPix is enabled
if "%1"=="--mathpix" (
  echo MathPix integration enabled
  set USE_MATHPIX=1
  python examples/worker_extract_enhanced.py --mode worker --workers 2 --use-mathpix
) else (
  echo MathPix integration disabled
  python examples/worker_extract_enhanced.py --mode worker --workers 2
)

:: If we get here, an error occurred
echo Worker exited unexpectedly
pause
