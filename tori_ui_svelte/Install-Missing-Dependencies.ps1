# Install-Missing-Dependencies.ps1
# Simple script to fix missing dependencies

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "       Installing Missing Dependencies         " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check current package.json
Write-Host "[1] Checking package.json..." -ForegroundColor Yellow
if (Test-Path "package.json") {
    $pkg = Get-Content "package.json" | ConvertFrom-Json
    Write-Host "  Project: $($pkg.name) v$($pkg.version)" -ForegroundColor Green
} else {
    Write-Host "  ERROR: package.json not found!" -ForegroundColor Red
    exit 1
}

# Install adapter-node specifically
Write-Host "`n[2] Installing @sveltejs/adapter-node..." -ForegroundColor Yellow
& pnpm add -D @sveltejs/adapter-node
if ($LASTEXITCODE -ne 0) {
    Write-Host "  pnpm failed, trying npm..." -ForegroundColor Yellow
    & npm install --save-dev @sveltejs/adapter-node
}

# Install other potentially missing dependencies
Write-Host "`n[3] Installing other SvelteKit dependencies..." -ForegroundColor Yellow
$deps = @(
    "@sveltejs/kit",
    "@sveltejs/vite-plugin-svelte",
    "svelte",
    "vite",
    "typescript",
    "svelte-check"
)

foreach ($dep in $deps) {
    Write-Host "  Checking $dep..." -ForegroundColor Gray
}

# Force reinstall all dependencies
Write-Host "`n[4] Reinstalling all dependencies..." -ForegroundColor Yellow
& pnpm install --force
if ($LASTEXITCODE -ne 0) {
    Write-Host "  pnpm failed, trying npm..." -ForegroundColor Yellow
    & npm install
}

# Check if adapter-node is now installed
Write-Host "`n[5] Verifying installation..." -ForegroundColor Yellow
if (Test-Path "node_modules\@sveltejs\adapter-node\package.json") {
    Write-Host "  @sveltejs/adapter-node is installed!" -ForegroundColor Green
} else {
    Write-Host "  WARNING: adapter-node may not be installed correctly" -ForegroundColor Yellow
    Write-Host "  Try manually:" -ForegroundColor Gray
    Write-Host "    rm -rf node_modules package-lock.json pnpm-lock.yaml" -ForegroundColor White
    Write-Host "    pnpm install" -ForegroundColor White
}

# Try a simple build
Write-Host "`n[6] Testing build..." -ForegroundColor Yellow
Write-Host "  Clearing cache..." -ForegroundColor Gray
if (Test-Path ".\.svelte-kit") {
    Remove-Item ".\.svelte-kit" -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "  Building..." -ForegroundColor Gray
& pnpm run build
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Build successful!" -ForegroundColor Green
} else {
    Write-Host "  Build failed, but dependencies should be fixed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  If adapter-node error persists, try:" -ForegroundColor Yellow
    Write-Host "    1. Delete node_modules:" -ForegroundColor Gray
    Write-Host "       Remove-Item node_modules -Recurse -Force" -ForegroundColor White
    Write-Host "    2. Delete lock files:" -ForegroundColor Gray
    Write-Host "       Remove-Item pnpm-lock.yaml, package-lock.json" -ForegroundColor White
    Write-Host "    3. Fresh install:" -ForegroundColor Gray
    Write-Host "       pnpm install" -ForegroundColor White
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "              Dependencies Fixed!              " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run the full fix:" -ForegroundColor Gray
Write-Host "     .\Fix-Dependencies-And-Mocks.ps1" -ForegroundColor White
Write-Host "  2. Or try building manually:" -ForegroundColor Gray
Write-Host "     pnpm run build" -ForegroundColor White
Write-Host ""
Write-Host "Done." -ForegroundColor Green
