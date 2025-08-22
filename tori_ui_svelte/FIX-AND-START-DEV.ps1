Write-Host "=== FIXING SVELTEKIT DEV SERVER ===" -ForegroundColor Cyan
Write-Host ""

# Navigate to the tori_ui_svelte directory
Set-Location D:\Dev\kha\tori_ui_svelte

Write-Host "[1/4] Installing missing adapter-node..." -ForegroundColor Yellow
pnpm add -D @sveltejs/adapter-node

Write-Host ""
Write-Host "[2/4] Installing all dependencies..." -ForegroundColor Yellow
pnpm install

Write-Host ""
Write-Host "[3/4] Running SvelteKit sync to generate files..." -ForegroundColor Yellow
pnpm exec svelte-kit sync

Write-Host ""
Write-Host "[4/4] Starting dev server..." -ForegroundColor Green
Write-Host "Server will start on http://localhost:5173" -ForegroundColor Cyan
Write-Host ""

# Start the dev server
pnpm dev