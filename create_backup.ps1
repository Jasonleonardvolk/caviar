# Backup script for enhanced_launcher.py (PowerShell version)

Write-Host "Creating backup of enhanced_launcher.py..." -ForegroundColor Cyan
Copy-Item -Path "enhanced_launcher.py" -Destination "enhanced_launcher.py.original_prajna_backup" -Force
Write-Host "Backup created: enhanced_launcher.py.original_prajna_backup" -ForegroundColor Green

Write-Host ""
Write-Host "To revert to the original (slow) Prajna API, run:" -ForegroundColor Yellow
Write-Host "  Copy-Item enhanced_launcher.py.original_prajna_backup enhanced_launcher.py -Force" -ForegroundColor White
Write-Host ""
Write-Host "Current configuration uses quick_api_server.py for faster startup." -ForegroundColor Green
