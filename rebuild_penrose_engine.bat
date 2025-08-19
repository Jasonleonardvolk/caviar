@echo off
echo ============================================================
echo REBUILDING PENROSE ENGINE WITH FULL IMPLEMENTATION
echo ============================================================

cd /d "C:\Users\jason\Desktop\tori\kha\concept_mesh\penrose_rs"

echo 🔧 Cleaning previous build...
cargo clean

echo 🔨 Building new Penrose engine...
maturin develop --release

if %ERRORLEVEL% EQU 0 (
    echo ✅ Build successful!
    echo 🧪 Running verification test...
    cd /d "C:\Users\jason\Desktop\tori\kha"
    python verify_penrose_import_chain.py
) else (
    echo ❌ Build failed with error code %ERRORLEVEL%
    echo Check the output above for compilation errors
)

pause
