@echo off
echo ===============================================
echo TORI Complete System with Chat Integration
echo ===============================================
echo.
echo Starting backend API with MCP integration...
echo.

REM Start the unified backend with MCP in a new window
start "TORI Unified Backend with MCP" cmd /k "cd /d C:\Users\jason\Desktop\tori\kha && python start_unified_tori.py"

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 8 /nobreak >nul

REM Start the frontend in a new window
start "TORI Frontend UI" cmd /k "cd /d C:\Users\jason\Desktop\tori\kha\tori_ui_svelte && npm run dev"

echo.
echo ===============================================
echo TORI System Starting...
echo ===============================================
echo.
echo Backend API: Starting on dynamic port (check first window)
echo Frontend UI: Starting on http://localhost:5173
echo.
echo NEW FEATURES:
echo - 🤖 Real chat API integration
echo - 🔍 Concept search functionality  
echo - 🌊 Soliton Memory integration
echo - 📊 Real-time confidence scoring
echo - ⏱️ Processing time tracking
echo - 🚀 MCP Server integration (NOW WORKING!)
echo - 🔧 Dynamic port management
echo - 📊 Status monitoring and error reporting
echo.
echo Test the chat by asking: "what do you know about darwin"
echo.
echo ===============================================
pause
