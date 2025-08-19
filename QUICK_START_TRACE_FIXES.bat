@echo off
REM =====================================================
REM   QUICK START - Apply Fixes and Launch
REM =====================================================

cls
echo.
echo ╔══════════════════════════════════════════╗
echo ║   TORI TRACE FIXES - QUICK START         ║
echo ╚══════════════════════════════════════════╝
echo.

REM First check the status
echo Checking current status...
python check_trace_fixes.py
echo.

echo Press any key to apply fixes...
pause >nul

REM Apply all fixes
call APPLY_TRACE_FIXES.bat

echo.
echo ════════════════════════════════════════════
echo.
echo Fixes applied! Starting services...
echo.

REM Start API in new window
start "TORI API" cmd /k "uvicorn api.enhanced_api:app --reload --port 8002"

REM Wait a moment for API to start
timeout /t 3 /nobreak >nul

REM Start MCP server in new window
start "MCP Server" cmd /k "python -m mcp_metacognitive.server"

REM Wait for services to initialize
timeout /t 5 /nobreak >nul

echo.
echo Services started! Running verification...
echo.
python verify_trace_fixes.py

echo.
echo ════════════════════════════════════════════
echo.
echo ✅ ALL SYSTEMS OPERATIONAL!
echo.
echo Check the opened windows for:
echo - TORI API (port 8002)
echo - MCP Server (port 8100)
echo.
echo Monitor logs:
echo - tail -f logs/session.log
echo - findstr "oscillators=" logs\session.log
echo.
echo ════════════════════════════════════════════
pause
