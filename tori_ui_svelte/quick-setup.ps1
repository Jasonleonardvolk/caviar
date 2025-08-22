# Quick Dev Setup Script
# One command to rule them all!

Write-Host "🚀 TORI Quick Dev Setup" -ForegroundColor Cyan

# Ensure we're in the frontend directory
cd tori_ui_svelte

# Install npm-run-all if not present
if (-not (Test-Path "node_modules\npm-run-all")) {
    Write-Host "Installing npm-run-all for concurrent execution..." -ForegroundColor Yellow
    npm install --save-dev npm-run-all
}

# Fix dependencies if needed
Write-Host "Fixing dependencies..." -ForegroundColor Yellow
npm run fix:deps

# Now you can simply run:
Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "`nTo start everything, just run:" -ForegroundColor Cyan
Write-Host "   npm run dev" -ForegroundColor White
Write-Host "`nThis will start both backend and frontend together!" -ForegroundColor Green
