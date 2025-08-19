# Install WebGPU Types
Write-Host "Installing @webgpu/types for TORI Project..." -ForegroundColor Cyan

# Save current directory
$originalDir = Get-Location

try {
    # Install in root
    Write-Host "`nInstalling in root directory..." -ForegroundColor Yellow
    Set-Location "D:\Dev\kha"
    npm install --save-dev @webgpu/types
    
    # Install in tori_ui_svelte
    Write-Host "`nInstalling in tori_ui_svelte..." -ForegroundColor Yellow
    Set-Location "D:\Dev\kha\tori_ui_svelte"
    npm install --save-dev @webgpu/types
    
    # Install in frontend
    Write-Host "`nInstalling in frontend..." -ForegroundColor Yellow
    Set-Location "D:\Dev\kha\frontend"
    npm install --save-dev @webgpu/types
    
    # Install in frontend/hybrid
    Write-Host "`nInstalling in frontend/hybrid..." -ForegroundColor Yellow
    Set-Location "D:\Dev\kha\frontend\hybrid"
    npm install --save-dev @webgpu/types
    
    Write-Host "`n✅ WebGPU types installed successfully!" -ForegroundColor Green
    
    # Test type checking
    Write-Host "`nTesting TypeScript compilation..." -ForegroundColor Cyan
    Set-Location "D:\Dev\kha"
    npx tsc --noEmit
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ TypeScript compilation successful - No errors!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ TypeScript compilation has some issues" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Error during installation: $_" -ForegroundColor Red
} finally {
    # Return to original directory
    Set-Location $originalDir
}

Write-Host "`nInstallation complete! You can now run:" -ForegroundColor Cyan
Write-Host "  npm run type-check" -ForegroundColor White
Write-Host "  npm run build" -ForegroundColor White
