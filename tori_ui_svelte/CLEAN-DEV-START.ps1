Write-Host "=== CLEAN DEV SERVER SETUP - FIRST TIME SUCCESS ===" -ForegroundColor Cyan
Write-Host ""

# Navigate to tori_ui_svelte
Set-Location D:\Dev\kha\tori_ui_svelte

Write-Host "[1/6] Killing any processes that might hold file locks..." -ForegroundColor Yellow
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process vite -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process tsserver -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "  [OK] Processes terminated" -ForegroundColor Green

Write-Host ""
Write-Host "[2/6] Removing conflicting node_modules and package manager files..." -ForegroundColor Yellow
Remove-Item -Recurse -Force .\node_modules -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\node_modules\.ignored -ErrorAction SilentlyContinue
Remove-Item -Force .\package-lock.json -ErrorAction SilentlyContinue
Remove-Item -Force .\yarn.lock -ErrorAction SilentlyContinue
Write-Host "  [OK] Old files removed" -ForegroundColor Green

Write-Host ""
Write-Host "[3/6] Activating pnpm as the package manager..." -ForegroundColor Yellow
corepack enable
corepack prepare pnpm@latest --activate
Write-Host "  [OK] pnpm activated" -ForegroundColor Green

Write-Host ""
Write-Host "[4/6] Installing fresh dependencies with pnpm..." -ForegroundColor Yellow
pnpm install
Write-Host "  [OK] Dependencies installed" -ForegroundColor Green

Write-Host ""
Write-Host "[5/6] Installing required adapters..." -ForegroundColor Yellow
pnpm add -D @sveltejs/adapter-node @sveltejs/adapter-auto
Write-Host "  [OK] Adapters installed" -ForegroundColor Green

Write-Host ""
Write-Host "[6/6] Generating SvelteKit files..." -ForegroundColor Yellow
pnpm exec svelte-kit sync
Write-Host "  [OK] SvelteKit synced" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " SETUP COMPLETE - Starting dev server" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Start the dev server
Write-Host "Starting dev server on http://localhost:5173" -ForegroundColor Cyan
pnpm dev -- --host 0.0.0.0 --port 5173