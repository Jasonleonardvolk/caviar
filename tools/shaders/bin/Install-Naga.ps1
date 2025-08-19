# Install-Naga.ps1
# Alternative: Install Naga if Tint is not available
# Naga is part of wgpu and can validate WGSL

$ErrorActionPreference = "Stop"

Write-Host "🦀 Installing Naga (WGSL Validator from wgpu)..." -ForegroundColor Cyan

# Check if Rust/Cargo is installed
try {
    $cargoVersion = cargo --version
    Write-Host "✅ Cargo found: $cargoVersion" -ForegroundColor Green
} catch {
    Write-Host @"
❌ Rust/Cargo not found!

To install Naga, you need Rust:
1. Go to: https://rustup.rs/
2. Download and run rustup-init.exe
3. Follow the installation prompts
4. Restart PowerShell
5. Run this script again

Alternative: Download pre-built Naga:
  https://github.com/gfx-rs/wgpu/releases
  Look for naga-cli or wgpu-tools
"@ -ForegroundColor Red
    exit 1
}

# Install naga-cli
Write-Host "Installing naga-cli via Cargo..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray

try {
    cargo install naga-cli --version 0.20.0
    
    # Find where cargo installed it
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { "$env:USERPROFILE\.cargo" }
    $nagaPath = Join-Path $cargoHome "bin\naga.exe"
    
    if (Test-Path $nagaPath) {
        # Copy to our bin directory
        $destination = Join-Path $PSScriptRoot "naga.exe"
        Copy-Item $nagaPath $destination -Force
        
        Write-Host "✅ Naga installed successfully!" -ForegroundColor Green
        Write-Host "   Location: $destination" -ForegroundColor Gray
        
        # Test it
        & $destination --version
        
        Write-Host @"

✅ Naga is ready to use!

Test with:
  .\naga.exe validate shader.wgsl
  .\naga.exe info shader.wgsl
  
Convert WGSL to SPIR-V:
  .\naga.exe shader.wgsl shader.spv

"@ -ForegroundColor Cyan
    } else {
        Write-Host "⚠️  Naga installed but not found at expected location" -ForegroundColor Yellow
        Write-Host "   Check: $cargoHome\bin\" -ForegroundColor Gray
    }
} catch {
    Write-Host "❌ Installation failed: $_" -ForegroundColor Red
    Write-Host @"

Alternative: Download pre-built binary:
1. Go to: https://github.com/gfx-rs/wgpu/releases
2. Download naga-cli for Windows
3. Extract naga.exe to: $PSScriptRoot
"@ -ForegroundColor Yellow
}
