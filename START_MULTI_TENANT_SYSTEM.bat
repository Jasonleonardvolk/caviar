@echo off
echo 🏢 TORI Multi-Tenant System - Quick Start
echo ==========================================
echo.

echo 📂 Changing to TORI directory...
cd /d C:\Users\jason\Desktop\tori\kha

echo 📋 Starting TORI Multi-Tenant API Server...
echo.

REM Start the API server in background
start "TORI Multi-Tenant API" python start_dynamic_api.py

echo ⏳ Waiting for server to start...
timeout /t 8 /nobreak >nul

echo.
echo 🧪 Running Multi-Tenant System Tests...
echo.

REM Run the test suite
python test_multi_tenant_system.py

echo.
echo 🎯 Multi-Tenant System Ready!
echo.
echo 📚 Available endpoints:
echo    • Health Check:    http://localhost:8002/health
echo    • API Docs:        http://localhost:8002/docs
echo    • System Stats:    http://localhost:8002/admin/system/stats
echo.
echo 📖 Documentation: MULTI_TENANT_COMPLETE_GUIDE.md
echo.
echo 💡 To manually start: 
echo    cd C:\Users\jason\Desktop\tori\kha
echo    python start_dynamic_api.py
echo.
echo Press any key to continue...
pause >nul
