@echo off
echo Testing MCP Server directly...
echo ==============================

cd C:\Users\jason\Desktop\tori\kha\mcp_metacognitive

REM Set the correct environment variables (without MCP_ prefix)
set TRANSPORT_TYPE=sse
set SERVER_HOST=0.0.0.0
set SERVER_PORT=8100

echo.
echo Environment variables set:
echo   TRANSPORT_TYPE=%TRANSPORT_TYPE%
echo   SERVER_HOST=%SERVER_HOST%
echo   SERVER_PORT=%SERVER_PORT%
echo.
echo Starting MCP server...
echo.

C:\ALANPY311\python.exe server.py
