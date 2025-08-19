@echo off
REM Quick deployment script for TORI/KHA (Windows)

echo 🚀 TORI/KHA Quick Deploy
echo ========================

REM Check if we're in the right directory
if not exist "start_tori.py" (
    echo ❌ Error: start_tori.py not found. Please run from project root.
    exit /b 1
)

echo 📁 Current directory: %CD%

REM Step 1: Fix imports
echo 🔧 Step 1: Fixing broken imports...
python import_fixer.py
echo ✅ Import fixes complete

REM Step 2: Install Python dependencies
echo 📦 Step 2: Installing Python dependencies...
pip --version >nul 2>&1
if %errorlevel% == 0 (
    pip install -r requirements.txt
    echo ✅ Python dependencies installed
) else (
    echo ⚠️  pip not found, skipping Python dependencies
)

REM Step 3: Install Node.js dependencies  
echo 📦 Step 3: Installing Node.js dependencies...
npm --version >nul 2>&1
if %errorlevel% == 0 (
    npm install
    echo ✅ Node.js dependencies installed
) else (
    echo ⚠️  npm not found, skipping Node.js dependencies
)

REM Step 4: Create data directories
echo 📁 Step 4: Creating data directories...
if not exist "data" mkdir data
if not exist "data\cognitive" mkdir data\cognitive
if not exist "data\memory_vault" mkdir data\memory_vault
if not exist "data\eigenvalue_monitor" mkdir data\eigenvalue_monitor
if not exist "data\lyapunov" mkdir data\lyapunov
if not exist "data\koopman" mkdir data\koopman
echo ✅ Data directories created

REM Step 5: Test Python imports
echo 🧪 Step 5: Testing Python imports...
python -c "try:\n    from python.core.CognitiveEngine import CognitiveEngine\n    from python.core.memory_vault import UnifiedMemoryVault\n    from python.stability.eigenvalue_monitor import EigenvalueMonitor\n    print('✅ All Python imports successful')\nexcept ImportError as e:\n    print(f'❌ Import error: {e}')\n    exit(1)"

if %errorlevel% neq 0 (
    echo ❌ Python import test failed
    echo 💡 Try: set PYTHONPATH=%PYTHONPATH%;%CD%\python
    exit /b 1
)

echo.
echo 🎉 TORI/KHA deployment complete!
echo.
echo 🚀 To start the system:
echo    python start_tori.py
echo.
echo 🌐 Frontend will be available at:
echo    http://localhost:5173
echo.
echo 📊 For troubleshooting, see:
echo    DEPLOYMENT_GUIDE.md
echo.

pause
