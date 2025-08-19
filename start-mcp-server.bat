@echo off
echo Starting MCP 2.0 Server...

rem Set environment variables
set PCC_WIRE_FORMAT=json
rem set ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

rem Start the server with optimized settings
echo Using optimized server settings (uvloop, httptools)
python -m uvicorn backend.routes.mcp.server:app --host 127.0.0.1 --port 8787 --reload --reload-dir backend/routes/mcp --loop uvloop --http httptools

rem If the server fails to start, display an error message
if %ERRORLEVEL% NEQ 0 (
  echo Error starting MCP server. Make sure all dependencies are installed:
  echo pip install fastapi uvicorn pydantic httptools uvloop
  pause
)
