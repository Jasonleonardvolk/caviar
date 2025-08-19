# Set MCP transport to SSE and run the server
$env:MCP_TRANSPORT_TYPE = "sse"
$env:MCP_SERVER_HOST = "0.0.0.0"
$env:MCP_SERVER_PORT = "8100"

Write-Host "Running MCP server with SSE transport..." -ForegroundColor Cyan
Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "  MCP_TRANSPORT_TYPE = sse" -ForegroundColor Gray
Write-Host "  MCP_SERVER_HOST = 0.0.0.0" -ForegroundColor Gray
Write-Host "  MCP_SERVER_PORT = 8100" -ForegroundColor Gray

# Run the server
C:\ALANPY311\python.exe server.py
