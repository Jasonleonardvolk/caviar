# Run-ResetAndShip.ps1
# Wrapper to ensure Reset-And-Ship.ps1 runs correctly

Write-Host "=== RUNNING RESET AND SHIP ===" -ForegroundColor Cyan
Write-Host ""

# Clear any PowerShell cache
Remove-Variable * -ErrorAction SilentlyContinue
$Error.Clear()

# Set location
Set-Location D:\Dev\kha\tori_ui_svelte

# Source the script directly to avoid cache issues
try {
    # Run with direct invocation
    powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\release\Reset-And-Ship.ps1"
} catch {
    Write-Host "Error running Reset-And-Ship: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Trying alternative approach..." -ForegroundColor Yellow
    
    # Alternative: run the build and start manually
    Write-Host "Building..." -ForegroundColor Cyan
    pnpm run build
    
    Write-Host "Starting server..." -ForegroundColor Cyan
    $env:PORT = "3000"
    $env:IRIS_USE_MOCKS = "1"
    $env:IRIS_ALLOW_UNAUTH = "1"
    
    Start-Job -Name "iris-server" -ScriptBlock {
        Set-Location "D:\Dev\kha\tori_ui_svelte"
        $env:PORT = "3000"
        $env:IRIS_USE_MOCKS = "1"
        $env:IRIS_ALLOW_UNAUTH = "1"
        node build/index.js
    }
    
    Write-Host ""
    Write-Host "Server started!" -ForegroundColor Green
    Write-Host "Access at: http://localhost:3000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  Get-Job iris-server     - Check status" -ForegroundColor White
    Write-Host "  Stop-Job iris-server    - Stop server" -ForegroundColor White
    Write-Host "  Remove-Job iris-server  - Clean up" -ForegroundColor White
}
