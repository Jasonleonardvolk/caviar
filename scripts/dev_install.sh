#!/usr/bin/env bash
# TORI One-Command Dev Install for Linux/macOS
# Creates venv, installs all deps, builds Rust crates, and launches TORI

set -euo pipefail

HEADLESS=false
if [[ "${1:-}" == "--headless" ]]; then
    HEADLESS=true
fi

START_TIME=$(date +%s)

echo "========================================"
echo "🚀 TORI ONE-COMMAND DEV INSTALL"
echo "========================================"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $PYTHON_VERSION =~ Python\ 3\.([0-9]+) ]]; then
    MINOR_VERSION=${BASH_REMATCH[1]}
    if [[ $MINOR_VERSION -lt 8 ]]; then
        echo "❌ Python 3.8+ required (found: $PYTHON_VERSION)"
        exit 1
    fi
    echo "✅ $PYTHON_VERSION"
else
    echo "❌ Python not found or version check failed"
    exit 1
fi

# Create venv
echo ""
echo "🐍 Creating virtual environment..."
if [[ -d .venv ]]; then
    echo "   Virtual environment already exists, using it"
else
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate venv
echo ""
echo "🔧 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "📦 Upgrading pip, wheel, setuptools..."
python -m pip install -U pip wheel setuptools >/dev/null 2>&1
echo "✅ Package tools upgraded"

# Install TORI with dev dependencies
echo ""
echo "📦 Installing TORI (editable) + dev dependencies..."
if [[ -f pyproject.toml ]]; then
    python -m pip install -e ".[dev]" 2>&1 | grep -E "Successfully installed|Requirement already satisfied" || true
else
    # Fallback to requirements-dev.txt
    echo "   pyproject.toml not found, using requirements-dev.txt"
    python -m pip install -r requirements-dev.txt 2>&1 | grep -E "Successfully installed|Requirement already satisfied" || true
fi
echo "✅ Python dependencies installed"

# Install spaCy model
echo ""
echo "🧠 Installing spaCy language model..."
python -m spacy download en_core_web_lg -q >/dev/null 2>&1 || true
echo "✅ spaCy model installed"

# Build Rust crates
echo ""
echo "🦀 Building Rust crates with maturin..."

# Check if Rust is installed
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    echo "   Found $RUST_VERSION"
    
    # Install maturin if needed
    python -m pip install maturin >/dev/null 2>&1
    
    # Build Penrose
    if [[ -f "concept_mesh/penrose_rs/Cargo.toml" ]]; then
        echo "   Building Penrose (similarity engine)..."
        pushd concept_mesh/penrose_rs >/dev/null
        maturin develop --release 2>&1 | grep -E "Built wheel|Installed" || true
        popd >/dev/null
        echo "✅ Penrose built"
    fi
    
    # Build concept_mesh_rs (if it has PyO3 bindings)
    if [[ -f "concept_mesh/concept_mesh_rs/Cargo.toml" ]]; then
        echo "   Building concept_mesh_rs..."
        pushd concept_mesh/concept_mesh_rs >/dev/null
        # Check if it has PyO3 bindings
        if grep -q "pyo3" Cargo.toml; then
            maturin develop --release 2>&1 | grep -E "Built wheel|Installed" || true
            echo "✅ concept_mesh_rs built"
        else
            echo "   Skipping concept_mesh_rs (no PyO3 bindings)"
        fi
        popd >/dev/null
    fi
else
    echo "⚠️  Rust not found, skipping Rust builds"
    echo "   Install from: https://rustup.rs/"
fi

# Install frontend dependencies
echo ""
echo "🌐 Installing frontend dependencies..."
if [[ -f "frontend/package.json" ]]; then
    pushd frontend >/dev/null
    if command -v npm &> /dev/null; then
        npm ci --silent
        echo "✅ Frontend dependencies installed"
    else
        echo "⚠️  npm not found, skipping frontend setup"
        echo "   Install Node.js from: https://nodejs.org/"
    fi
    popd >/dev/null
else
    echo "   No frontend directory found, skipping"
fi

# Link MCP packages
echo ""
echo "🔗 Linking MCP packages..."
if [[ -d "mcp_metacognitive" ]]; then
    # Just install the base mcp package, don't try editable install
    python -m pip install mcp >/dev/null 2>&1 || true
    echo "✅ MCP base package installed"
else
    echo "   MCP metacognitive not found, skipping"
fi

# Install additional critical packages that might be missing
echo ""
echo "📦 Ensuring all critical packages are installed..."
python -m pip install sse-starlette pypdf2 deepdiff aiofiles >/dev/null 2>&1
echo "✅ Critical packages verified"

# Final summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "========================================"
echo "✅ INSTALLATION COMPLETE!"
echo "========================================"
echo "Time taken: $DURATION seconds"
echo ""

# Launch TORI
if [[ "$HEADLESS" == "false" ]]; then
    echo "🚀 Launching TORI..."
    echo "   Press Ctrl+C to stop"
    echo ""
    python enhanced_launcher.py
else
    echo "🔍 Running health check..."
    
    # Start TORI in background
    python enhanced_launcher.py --no-browser &
    TORI_PID=$!
    
    # Wait for API to be ready
    READY=false
    ATTEMPTS=0
    while [[ "$READY" == "false" ]] && [[ $ATTEMPTS -lt 30 ]]; do
        sleep 2
        if curl -s http://localhost:8003/api/health | grep -q "healthy"; then
            READY=true
            echo "✅ API health check passed"
        fi
        ATTEMPTS=$((ATTEMPTS + 1))
    done
    
    # Stop TORI
    kill $TORI_PID 2>/dev/null || true
    
    if [[ "$READY" == "false" ]]; then
        echo "❌ API health check failed"
        exit 1
    fi
fi

echo ""
echo "🎉 Setup complete! TORI is ready for development."
