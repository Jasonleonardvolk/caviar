# Launch TORI with CORRECT MCP environment variables
Write-Host "`n🚀 LAUNCHING TORI WITH CORRECT MCP CONFIGURATION" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Set environment variables for MCP SSE transport (WITHOUT MCP_ prefix!)
Write-Host "`nSetting MCP environment variables..." -ForegroundColor Yellow

# These are the CORRECT variable names that config.py expects
$env:TRANSPORT_TYPE = "sse"
$env:SERVER_HOST = "0.0.0.0"
$env:SERVER_PORT = "8100"

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
Write-Host "`nThe MCP server should now start with SSE transport!" -ForegroundColor Yellow
Write-Host "Look for: 'Starting TORI MCP server with SSE transport on 0.0.0.0:8100...'" -ForegroundColor Yellow

python enhanced_launcher.py
