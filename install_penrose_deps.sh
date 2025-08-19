# Install Penrose Dependencies

echo "========================================"
echo "   Installing Penrose Dependencies"
echo "========================================"
echo ""

echo "[1/3] Installing scipy (sparse matrix operations)..."
python -m pip install scipy
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install scipy"
    exit 1
fi
echo "SUCCESS: scipy installed"
echo ""

echo "[2/3] Installing zstandard (compression)..."
python -m pip install zstandard
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install zstandard"
    exit 1
fi
echo "SUCCESS: zstandard installed"
echo ""

echo "[3/3] Installing numba (JIT compilation - optional but recommended)..."
python -m pip install numba
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to install numba (optional - Penrose will work without it)"
    echo "         You'll still get great performance, just not the maximum possible"
else
    echo "SUCCESS: numba installed (2x additional speedup enabled!)"
fi
echo ""

echo "========================================"
echo "   Verifying Penrose Installation"
echo "========================================"
echo ""

python verify_penrose.py

echo ""
echo "========================================"
echo "   Installation Complete!"
echo "========================================"
echo ""
echo "Next step: Restart TORI with"
echo "  python enhanced_launcher.py"
echo ""
