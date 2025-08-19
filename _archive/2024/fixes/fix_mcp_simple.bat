@echo off
echo ========================================
echo FIX: MCP Simple Setup (Without editable)
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo [1/2] Installing MCP base package only...
pip install mcp

echo.
echo [2/2] Testing MCP imports...
echo.

echo Testing MCP types...
python -c "from mcp.types import Tool, TextContent, Resource; print('✅ MCP types OK')"

echo.
echo Testing MCP server...
python -c "from mcp.server import Server; print('✅ MCP server OK')"

echo.
echo ========================================
echo ✅ MCP Base Package Installed!
echo ========================================
echo.
echo The MCP types are now available.
echo The metacognitive server will use the files directly (not installed).
echo.
echo To start TORI without MCP metacognitive errors:
echo python enhanced_launcher.py --no-browser
echo.
pause
