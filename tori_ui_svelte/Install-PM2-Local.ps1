# Install-PM2-Local.ps1
# Helper script to install PM2 locally in the project

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "       Installing PM2 Locally          " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if PM2 is already available
$pm2Global = Get-Command pm2 -ErrorAction SilentlyContinue
if ($pm2Global) {
    Write-Host "PM2 is already installed globally at:" -ForegroundColor Green
    Write-Host "  $($pm2Global.Source)" -ForegroundColor Gray
    Write-Host ""
}

# Check if PM2 is in local node_modules
if (Test-Path "node_modules\pm2") {
    Write-Host "PM2 is already installed locally!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can use PM2 with:" -ForegroundColor Yellow
    Write-Host "  npx pm2 <command>" -ForegroundColor Gray
} else {
    Write-Host "Installing PM2 locally..." -ForegroundColor Yellow
    & npm install pm2
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "PM2 installed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now use PM2 with:" -ForegroundColor Yellow
        Write-Host "  npx pm2 <command>" -ForegroundColor Gray
    } else {
        Write-Host ""
        Write-Host "ERROR: Failed to install PM2" -ForegroundColor Red
        Write-Host "Try running: npm install pm2" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "Common PM2 commands:" -ForegroundColor Cyan
Write-Host "  npx pm2 start build/index.js --name iris    # Start the app" -ForegroundColor Gray
Write-Host "  npx pm2 logs iris                           # View logs" -ForegroundColor Gray
Write-Host "  npx pm2 status                              # Check status" -ForegroundColor Gray
Write-Host "  npx pm2 restart iris                        # Restart app" -ForegroundColor Gray
Write-Host "  npx pm2 stop iris                           # Stop app" -ForegroundColor Gray
Write-Host "  npx pm2 delete iris                         # Remove from PM2" -ForegroundColor Gray
Write-Host "  npx pm2 monit                               # Monitor dashboard" -ForegroundColor Gray
Write-Host ""
Write-Host "Done!" -ForegroundColor Green
