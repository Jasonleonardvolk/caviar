@echo off
echo ========================================
echo FIX: MCP Metacognitive Import
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing MCP metacognitive in editable mode...
cd mcp_metacognitive
pip install -e .
cd ..

echo.
echo Testing MCP imports...
python -c "from mcp.types import Tool, TextContent; print('✅ MCP types OK')"
python -c "from mcp_metacognitive import server; print('✅ MCP metacognitive server OK')"

echo.
echo ✅ MCP metacognitive installed!
echo.
echo Alternative: If installation fails, the launcher already has a fallback.
echo.
pause
