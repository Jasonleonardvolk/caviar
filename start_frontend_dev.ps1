Write-Host "`n🚀 Starting TORI Frontend Development Server" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

Set-Location -Path "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte"

# Clear any old build artifacts
if (Test-Path ".svelte-kit") {
    Write-Host "Clearing build cache..." -ForegroundColor Yellow
    Remove-Item -Path ".svelte-kit" -Recurse -Force
}

Write-Host "`nStarting development server..." -ForegroundColor Green
Write-Host "If you see any errors, press Ctrl+C and we'll fix them.`n" -ForegroundColor Yellow

# Start the dev server
npm run dev
