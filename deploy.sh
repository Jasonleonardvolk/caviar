#!/bin/bash
# Quick deployment script for TORI/KHA

echo "🚀 TORI/KHA Quick Deploy"
echo "========================"

# Check if we're in the right directory
if [ ! -f "start_tori.py" ]; then
    echo "❌ Error: start_tori.py not found. Please run from project root."
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Step 1: Fix imports
echo "🔧 Step 1: Fixing broken imports..."
python import_fixer.py
echo "✅ Import fixes complete"

# Step 2: Install Python dependencies
echo "📦 Step 2: Installing Python dependencies..."
if command -v pip &> /dev/null; then
    pip install -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️  pip not found, skipping Python dependencies"
fi

# Step 3: Install Node.js dependencies  
echo "📦 Step 3: Installing Node.js dependencies..."
if command -v npm &> /dev/null; then
    npm install
    echo "✅ Node.js dependencies installed"
else
    echo "⚠️  npm not found, skipping Node.js dependencies"
fi

# Step 4: Create data directories
echo "📁 Step 4: Creating data directories..."
mkdir -p data/cognitive
mkdir -p data/memory_vault
mkdir -p data/eigenvalue_monitor
mkdir -p data/lyapunov
mkdir -p data/koopman
echo "✅ Data directories created"

# Step 5: Test Python imports
echo "🧪 Step 5: Testing Python imports..."
python -c "
try:
    from python.core.CognitiveEngine import CognitiveEngine
    from python.core.memory_vault import UnifiedMemoryVault
    from python.stability.eigenvalue_monitor import EigenvalueMonitor
    print('✅ All Python imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
" || {
    echo "❌ Python import test failed"
    echo "💡 Try: export PYTHONPATH=\$PYTHONPATH:\$(pwd)/python"
    exit 1
}

echo ""
echo "🎉 TORI/KHA deployment complete!"
echo ""
echo "🚀 To start the system:"
echo "   python start_tori.py"
echo ""
echo "🌐 Frontend will be available at:"
echo "   http://localhost:5173"
echo ""
echo "📊 For troubleshooting, see:"
echo "   DEPLOYMENT_GUIDE.md"
echo ""
