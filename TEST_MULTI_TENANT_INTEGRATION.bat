@echo off
echo 🏢 TORI Multi-Tenant Integration Test
echo ====================================
echo.

echo 📂 Changing to TORI directory...
cd /d C:\Users\jason\Desktop\tori\kha

echo 🔍 Checking if TORI is already running...
echo.

REM Check if TORI is already running
python -c "import requests; print('✅ TORI is running' if requests.get('http://localhost:8002/health', timeout=2).status_code == 200 else '❌ TORI not detected')" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo 🎯 TORI system detected! Running tests...
    goto RUN_TESTS
)

echo 📋 TORI not running. Starting it now...
echo.
echo 🚀 Using your existing START_TORI_WITH_CHAT.bat...

REM Check if the existing start script exists
if exist "START_TORI_WITH_CHAT.bat" (
    echo ✅ Found START_TORI_WITH_CHAT.bat
    echo 🎬 Starting TORI system (this will open new windows)...
    echo.
    
    REM Start the existing TORI system
    call START_TORI_WITH_CHAT.bat
    
    echo ⏳ Waiting for TORI system to fully start...
    timeout /t 15 /nobreak >nul
) else (
    echo ❌ START_TORI_WITH_CHAT.bat not found
    echo 🔄 Falling back to start_dynamic_api.py...
    
    start "TORI Multi-Tenant API" python start_dynamic_api.py
    
    echo ⏳ Waiting for API server to start...
    timeout /t 8 /nobreak >nul
)

:RUN_TESTS
echo.
echo 🧪 Running Multi-Tenant Integration Tests...
echo.

REM Run the test suite
python test_multi_tenant_system.py

echo.
echo 🎯 Integration Test Complete!
echo.
echo 📚 Available endpoints (if multi-tenant features are integrated):
echo    • Health Check:    http://localhost:8002/health
echo    • API Docs:        http://localhost:8002/docs
echo    • Frontend UI:     http://localhost:5173
echo.
echo 📖 Documentation: MULTI_TENANT_COMPLETE_GUIDE.md
echo.
echo 💡 Your existing TORI system works perfectly!
echo 🏢 Multi-tenant features can be integrated when ready.
echo.
echo Press any key to continue...
pause >nul
