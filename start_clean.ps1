# TORI Clean Start Script
Write-Host "🚀 Starting TORI System (Clean)" -ForegroundColor Cyan

# Step 1: Kill any existing processes
Write-Host "`nStep 1: Cleaning up old processes..." -ForegroundColor Yellow
python kill_ports.py

# Step 2: Clear SvelteKit cache
Write-Host "`nStep 2: Clearing SvelteKit cache..." -ForegroundColor Yellow
cd tori_ui_svelte
if (Test-Path ".svelte-kit") {
    Remove-Item .svelte-kit -Recurse -Force
    Write-Host "✅ Cache cleared" -ForegroundColor Green
}

# Step 3: Go back to main directory
cd ..

# Step 4: Start the system
Write-Host "`nStep 3: Starting TORI system..." -ForegroundColor Yellow
poetry run python enhanced_launcher.py

Write-Host "`n✅ TORI system started!" -ForegroundColor Green
