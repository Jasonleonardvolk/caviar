# Launch TORI with proper MCP configuration
Write-Host "`n🚀 LAUNCHING TORI WITH PROPER MCP CONFIGURATION" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Set environment variables for MCP SSE transport
Write-Host "`nSetting MCP environment variables..." -ForegroundColor Yellow

$env:TRANSPORT_TYPE = "sse"
$env:SERVER_HOST = "0.0.0.0"
$env:SERVER_PORT = "8100"
$env:MCP_TRANSPORT_TYPE = "sse"
$env:MCP_SERVER_HOST = "0.0.0.0"
$env:MCP_SERVER_PORT = "8100"

Write-Host "✅ Environment variables set:" -ForegroundColor Green
Write-Host "   TRANSPORT_TYPE = sse" -ForegroundColor Gray
Write-Host "   SERVER_HOST = 0.0.0.0" -ForegroundColor Gray
Write-Host "   SERVER_PORT = 8100" -ForegroundColor Gray

# Change to TORI directory
Write-Host "`nChanging to TORI directory..." -ForegroundColor Yellow
Set-Location "C:\Users\jason\Desktop\tori\kha"

# Launch TORI
Write-Host "`n🎯 Launching TORI Enhanced Launcher..." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

python enhanced_launcher.py
