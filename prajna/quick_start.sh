#!/bin/bash
"""
Prajna Quick Start Script
========================

This script sets up and launches Prajna with minimal configuration.
Perfect for getting started quickly or testing the system.
"""

echo "🧠 Prajna Quick Start - TORI's Voice and Language Model"
echo "======================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if (( $(echo "$python_version >= 3.8" | bc -l) )); then
    echo "✅ Python version: $(python3 --version)"
else
    echo "❌ Python 3.8+ required. Found: $(python3 --version)"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet fastapi uvicorn websockets aiohttp numpy scikit-learn

# Optional dependencies (install if available)
echo "🔧 Installing optional dependencies..."
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || echo "⚠️  PyTorch not installed (using fallback)"
pip install --quiet transformers 2>/dev/null || echo "⚠️  Transformers not installed (using fallback)"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models logs snapshots

# Run tests
echo "🧪 Running Prajna tests..."
python3 test_prajna.py
test_result=$?

if [ $test_result -eq 0 ]; then
    echo ""
    echo "🎉 All tests passed! Starting Prajna in demo mode..."
    echo ""
    echo "🔗 API will be available at: http://localhost:8001"
    echo "🔗 Health check: http://localhost:8001/api/health"
    echo "🔗 API docs: http://localhost:8001/docs"
    echo ""
    echo "📝 To test Prajna:"
    echo "   curl -X POST http://localhost:8001/api/answer \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"user_query\": \"What is Prajna?\"}'"
    echo ""
    echo "🛑 Press Ctrl+C to stop Prajna"
    echo ""
    
    # Start Prajna
    python3 start_prajna.py --demo --log-level INFO
else
    echo "❌ Tests failed! Please check the output above."
    exit 1
fi
