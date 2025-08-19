#!/bin/bash
# Start-Ingest-Bus.sh
# 
# This script starts the TORI Ingest Bus service with proper environment setup.
# It ensures dependencies are available and configures logging.

echo "====================================="
echo "  TORI Ingest Bus Service Launcher"
echo "====================================="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check for required Python packages
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, pydantic" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install fastapi uvicorn pydantic prometheus-client
fi

# Ensure log directory exists
mkdir -p logs/ingest_bus

# Start the service
echo "Starting Ingest Bus service on port 8080..."
cd ingest_bus
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8080 --log-level info > ../logs/ingest_bus/service.log 2>&1 &

# Save PID for later management
echo $! > ../logs/ingest_bus/service.pid
echo "Service started with PID: $(cat ../logs/ingest_bus/service.pid)"
echo "Logs available at: $(pwd)/../logs/ingest_bus/service.log"
echo "API available at: http://localhost:8080"
echo "Documentation at: http://localhost:8080/docs"
echo "====================================="
