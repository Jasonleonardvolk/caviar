# TORI Chat Launcher for PowerShell
# Easy-to-use launcher that handles all the deployment steps

Write-Host @"
==========================================
    Starting TORI Chat with MCP
==========================================
"@ -ForegroundColor Cyan

# Change to the correct directory
Set-Location "C:\Users\jason\Desktop\tori\kha"

# Check if we should use the PowerShell or batch version
$usePowerShell = $true

if ($usePowerShell) {
    Write-Host "Using PowerShell deployment script..." -ForegroundColor Yellow
    & .\deploy-tori-chat-with-mcp.ps1
} else {
    Write-Host "Using batch deployment script..." -ForegroundColor Yellow
    & cmd.exe /c "deploy-tori-chat-with-mcp.bat"
}

# Keep window open if there was an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPress any key to exit..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
