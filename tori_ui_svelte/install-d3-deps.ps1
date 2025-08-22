# Install D3 dependencies for ConceptGraph component
Write-Host "Installing D3 dependencies for ConceptGraph component..." -ForegroundColor Cyan
Write-Host ""

Set-Location -Path "D:\Dev\kha\tori_ui_svelte"

Write-Host "Installing d3..." -ForegroundColor Yellow
pnpm add d3

Write-Host ""
Write-Host "Installing @types/d3 as dev dependency..." -ForegroundColor Yellow  
pnpm add -D @types/d3

Write-Host ""
Write-Host "D3 dependencies installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
