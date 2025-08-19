#!/bin/bash
# Launch script for TORI Chaos-Enhanced System

echo "🚀 Launching TORI Chaos-Enhanced System..."
echo "=================================="

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CHAOS_EXPERIMENT=1

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Installing requirements..."
    pip install numpy scipy numba pytest websockets pyyaml matplotlib
fi

# Create necessary directories
mkdir -p logs
mkdir -p snapshots
mkdir -p conf

# Check if config files exist
if [ ! -f "conf/lattice_config.yaml" ]; then
    echo "Warning: lattice_config.yaml not found"
fi

# Launch options
case "${1:-full}" in
    "full")
        echo "Starting full system..."
        python3 tori_master.py
        ;;
    "test")
        echo "Running integration tests..."
        python3 -m pytest integration_tests.py -v
        ;;
    "monitor")
        echo "Starting health monitor..."
        python3 system_health_monitor.py
        ;;
    "websocket")
        echo "Starting WebSocket server only..."
        python3 services/metrics_ws.py
        ;;
    "demo")
        echo "Running demo..."
        python3 -c "
import asyncio
from tori_master import TORIMaster

async def demo():
    master = TORIMaster()
    await master.start()
    
    # Run some demo queries
    queries = [
        'What is consciousness?',
        'Explore patterns in quantum systems',
        'Create a novel approach to AI safety'
    ]
    
    for q in queries:
        print(f'\nQuery: {q}')
        result = await master.process_query(q, {'enable_chaos': True})
        print(f'Response: {result[\"response\"][:200]}...')
        
    await master.stop()

asyncio.run(demo())
"
        ;;
    *)
        echo "Usage: $0 [full|test|monitor|websocket|demo]"
        exit 1
        ;;
esac

echo "=================================="
echo "TORI shutdown complete"
