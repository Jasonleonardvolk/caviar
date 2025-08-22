# PowerShell Script to Setup iRis Paywall
Write-Host "Installing Stripe SDK..." -ForegroundColor Green
Set-Location D:\Dev\kha\tori_ui_svelte
npm install stripe

Write-Host "`niRis Setup Complete!" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Set up your Stripe account at https://dashboard.stripe.com"
Write-Host "2. Create test products and prices"
Write-Host "3. Update .env with your actual Stripe keys"
Write-Host "4. Run: npm run dev"
Write-Host "5. Visit: http://localhost:5173/hologram-studio"
Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")