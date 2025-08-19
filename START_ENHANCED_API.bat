@echo off
echo 🚀 STARTING TORI ENHANCED API SERVER
echo =====================================
echo.
echo 🎯 This will start the TORI API with new endpoints:
echo    - /multiply - Hyperbolic matrix multiplication
echo    - /intent - Intent-driven reasoning  
echo    - /ws/stability - Real-time stability monitoring
echo    - /ws/chaos - Chaos event streaming
echo.
echo 📍 API will be available at: http://localhost:8002
echo 🔗 Health check: http://localhost:8002/health/extended
echo 🌐 WebSocket test: http://localhost:8002/ws/test
echo 📄 API docs: http://localhost:8002/docs
echo 🛑 To stop: Press Ctrl+C in this window
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo 🐍 Starting Enhanced API server...
echo.

python main_enhanced.py

echo.
echo 🛑 Enhanced API server stopped.
echo 📝 To restart, run this batch file again.
pause
