@echo off
echo ========================================
echo Penrose Engine Local Sanity Check (Focused)
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "concept_mesh\penrose_rs\Cargo.toml" (
    echo ERROR: Run this script from the project root directory
    echo Expected location: C:\Users\jason\Desktop\tori\kha
    exit /b 1
)

cd concept_mesh\penrose_rs

echo [1/3] Checking Rust compilation (penrose_rs only)...
cargo check
if errorlevel 1 (
    echo ERROR: Rust compilation failed!
    cd ..\..
    exit /b 1
)
echo OK: Rust compiles successfully
echo.

echo [2/3] Building and installing wheel...
maturin develop --release
if errorlevel 1 (
    echo ERROR: Maturin build failed!
    cd ..\..
    exit /b 1
)
echo OK: Wheel built and installed
echo.

echo [3/3] Testing Python import...
cd ..\..
python -c "import penrose_engine_rs; print('OK: Rust import successful')"
if errorlevel 1 (
    echo ERROR: Python import failed!
    exit /b 1
)

echo.
echo ========================================
echo ALL CHECKS PASSED! Ready to commit and push.
echo ========================================
echo.
echo Next steps:
echo 1. git add -A
echo 2. git commit -m "feat: add Penrose Rust engine with CI"
echo 3. git push
echo 4. Check GitHub Actions for green CI build
echo.
pause
