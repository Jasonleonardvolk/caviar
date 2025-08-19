@echo off
echo ========================================
echo Bulletproof Penrose Build
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "concept_mesh\penrose_rs\Cargo.toml" (
    echo ERROR: Run this script from the project root directory
    echo Expected location: C:\Users\jason\Desktop\tori\kha
    exit /b 1
)

REM Step 1: Create venv if needed
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Step 2: Activate venv
echo Activating virtual environment...
call .venv\Scripts\activate

REM Step 3: Verify we're using the right Python
echo.
echo Checking Python interpreter...
where python
python -c "import sys; print('Using Python:', sys.executable)"

REM Step 4: Install maturin if needed
echo.
echo Checking maturin installation...
pip show maturin >nul 2>&1
if errorlevel 1 (
    echo Installing maturin...
    pip install maturin
)

REM Step 5: Build with explicit interpreter path
echo.
echo Building Penrose engine...
cd concept_mesh\penrose_rs
maturin develop --release -i ..\..\.venv\Scripts\python.exe

REM Step 6: Test import
echo.
echo Testing import...
cd ..\..
.venv\Scripts\python -c "import penrose_engine_rs as p; print('SUCCESS: Penrose Rust backend ready -', p.__name__)"

if errorlevel 0 (
    echo.
    echo ========================================
    echo BUILD SUCCESSFUL!
    echo ========================================
    echo The ModuleNotFoundError is fixed!
    echo.
    echo You can now:
    echo 1. git add/commit/push
    echo 2. Watch CI turn green
    echo.
) else (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo Check the error messages above.
    echo.
)

pause
