# TORI Fix WebGPU and Start Script
Write-Host "🚀 Fixing WebGPU Shader Constants and Starting TORI" -ForegroundColor Cyan

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

# Step 3: Clear any node_modules cache
Write-Host "`nStep 3: Clearing node_modules cache..." -ForegroundColor Yellow
if (Test-Path "node_modules\.cache") {
    Remove-Item node_modules\.cache -Recurse -Force
    Write-Host "✅ Node modules cache cleared" -ForegroundColor Green
}

# Step 4: Go back to main directory
cd ..

# Step 5: Start the system
Write-Host "`nStep 4: Starting TORI system with fixed WebGPU shaders..." -ForegroundColor Yellow
poetry run python enhanced_launcher.py

Write-Host "`n✅ TORI system started with WebGPU fixes!" -ForegroundColor Green
