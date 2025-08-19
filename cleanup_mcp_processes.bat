@echo off
echo Cleaning up MCP-related node processes...

:: Kill any existing MCP server processes
taskkill /F /IM node.exe /FI "WINDOWTITLE eq @modelcontextprotocol*" 2>nul
taskkill /F /IM node.exe /FI "COMMANDLINE eq *@modelcontextprotocol*" 2>nul

:: Wait a moment for processes to terminate
timeout /t 2 /nobreak >nul

:: Show remaining node processes for verification
echo.
echo Remaining node processes:
tasklist /FI "IMAGENAME eq node.exe" 2>nul

echo.
echo MCP process cleanup completed.
echo You can now restart Claude Desktop safely.
pause
