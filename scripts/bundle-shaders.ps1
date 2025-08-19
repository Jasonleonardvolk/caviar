# Bundle Shaders Script
# Run this to regenerate the shader bundle after making changes to WGSL files

Write-Host "Bundling WebGPU shaders..." -ForegroundColor Cyan

# Navigate to the project root
Push-Location $PSScriptRoot\..

try {
    # Run the TypeScript bundler
    npx tsx scripts/bundleShaders.ts
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nShader bundling completed successfully!" -ForegroundColor Green
        Write-Host "Generated: frontend/lib/webgpu/generated/shaderSources.ts" -ForegroundColor Yellow
    } else {
        Write-Host "`nShader bundling failed!" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}
