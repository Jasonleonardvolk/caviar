@echo off
echo ========================================
echo FIX: Install MCP Dependencies Directly
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing MCP and related packages directly...
pip install mcp>=1.0.0 fastmcp>=0.1.0

echo.
echo Installing other MCP metacognitive dependencies...
pip install aiofiles python-dotenv pydantic httpx uvicorn fastapi celery redis flower psutil

echo.
echo Testing imports...
python -c "from mcp.types import Tool, TextContent; print('✅ MCP types OK')"
python -c "from mcp.server import Server; print('✅ MCP server OK')"

echo.
echo ========================================
echo ✅ Dependencies Installed!
echo ========================================
echo.
echo The launcher will load MCP metacognitive from the directory.
echo No installation needed for the local package.
echo.
pause
