@echo off
echo 🚀 STARTING TORI DYNAMIC API SERVER
echo =====================================
echo.
echo 🎯 This will automatically find an available port
echo 📍 Port configuration will be saved for SvelteKit
echo 🔧 Handles port conflicts automatically
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo 🐍 Starting dynamic Python API server...
echo.

python start_dynamic_api.py

echo.
echo 🛑 Dynamic API server stopped.
pause
