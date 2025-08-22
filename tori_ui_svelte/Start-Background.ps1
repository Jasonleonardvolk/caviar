# Start-Background.ps1 - Starts mock server in background

Write-Host "Starting Mock Server in Background..." -ForegroundColor Cyan

# Kill existing node processes
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Remove old job if exists
Stop-Job -Name "iris-mock" -ErrorAction SilentlyContinue
Remove-Job -Name "iris-mock" -ErrorAction SilentlyContinue

# Start in background
$job = Start-Job -Name "iris-mock" -ScriptBlock {
    Set-Location D:\Dev\kha\tori_ui_svelte
    $env:IRIS_USE_MOCKS = "1"
    $env:IRIS_ALLOW_UNAUTH = "1"
    $env:PORT = "3000"
    node build\index.js
}

Write-Host "Server starting in background..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Check if it started
if ((Get-Job -Name "iris-mock").State -eq "Running") {
    Write-Host ""
    Write-Host "✓ Mock server running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access at: http://localhost:3000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  Get-Job iris-mock       - Check status" -ForegroundColor White
    Write-Host "  Receive-Job iris-mock   - View logs" -ForegroundColor White
    Write-Host "  Stop-Job iris-mock      - Stop server" -ForegroundColor White
    Write-Host "  Remove-Job iris-mock    - Clean up" -ForegroundColor White
    
    # Open browser
    Start-Process "http://localhost:3000"
} else {
    Write-Host "Failed to start. Check logs with: Receive-Job iris-mock" -ForegroundColor Red
}
