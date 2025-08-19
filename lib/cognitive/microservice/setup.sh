#!/bin/bash

# 🧠 TORI Cognitive Engine Microservice Setup Script
# Sets up both Node.js microservice and Python FastAPI bridge

echo "🧠 Setting up TORI Cognitive Engine Microservice..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Set up Node.js microservice
echo "📦 Setting up Node.js cognitive microservice..."
npm install

# Build TypeScript
echo "🔨 Building TypeScript..."
npx tsc

# Set up Python virtual environment
echo "🐍 Setting up Python FastAPI bridge..."
python -m venv venv

# Activate virtual environment (cross-platform)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install Python dependencies
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the services:"
echo "1. Node.js microservice: npm run dev (or npm start)"
echo "2. Python FastAPI bridge: python cognitive_bridge.py"
echo ""
echo "📊 Service endpoints:"
echo "- Node.js: http://localhost:4321/api/"
echo "- Python: http://localhost:8000/api/"
echo "- API docs: http://localhost:8000/docs"
