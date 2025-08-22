# Install missing PostCSS dependencies
Write-Host "Installing missing PostCSS dependencies..." -ForegroundColor Yellow

cd tori_ui_svelte

# Install autoprefixer and ensure PostCSS dependencies are correct
npm install --save-dev autoprefixer

Write-Host "✅ PostCSS dependencies installed!" -ForegroundColor Green
