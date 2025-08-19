@echo off
echo ========================================
echo FIX: MCP Metacognitive Complete Setup
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo [1/3] Installing MCP base package...
pip install mcp

echo.
echo [2/3] Installing MCP metacognitive in editable mode...
cd mcp_metacognitive
pip install -e .
cd ..

echo.
echo [3/3] Testing imports...
echo.

echo Testing MCP types...
python -c "from mcp.types import Tool, TextContent, Resource; print('✅ MCP types OK')"

echo.
echo Testing MCP server...
python -c "from mcp.server import Server; print('✅ MCP server OK')"

echo.
echo Testing MCP metacognitive...
python -c "import sys; sys.path.insert(0, 'mcp_metacognitive'); from core.config import config; print('✅ MCP metacognitive config OK')"

echo.
echo ========================================
echo ✅ MCP Setup Complete!
echo ========================================
echo.
echo The MCP metacognitive server is now properly installed.
echo TORI should now be able to start the cognitive engine.
echo.
pause
