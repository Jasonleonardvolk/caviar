@echo off
echo 🚀 STARTING TORI CONCEPT EXTRACTION API v2.0
echo =============================================
echo.
echo 🎯 This will start the Python concept extraction API
echo 📍 API will be available at: http://localhost:8002
echo 🔗 Health check: http://localhost:8002/health  
echo 📄 API docs: http://localhost:8002/docs
echo 🛑 To stop: Press Ctrl+C in this window
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo 🐍 Starting Python API server...
echo.

python main.py

echo.
echo 🛑 Python API server stopped.
echo 📝 To restart, run this batch file again.
pause
