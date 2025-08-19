# test_naga.ps1
# Quick test to verify naga installation and basic shader validation

Write-Host "`n==== NAGA SHADER VALIDATOR TEST ====" -ForegroundColor Cyan
Write-Host "Testing naga installation and basic validation`n" -ForegroundColor White

# Test 1: Check if naga is installed
Write-Host "[1] Checking naga installation..." -ForegroundColor Yellow
try {
    $nagaVersion = naga --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✓ Naga is installed: $nagaVersion" -ForegroundColor Green
    } else {
        Write-Host "    ✗ Naga not found. Installing..." -ForegroundColor Red
        Write-Host "    Running: cargo install naga-cli" -ForegroundColor Gray
        cargo install naga-cli
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    ✗ Failed to install naga." -ForegroundColor Red
            Write-Host "    Please ensure Rust and Cargo are installed:" -ForegroundColor Yellow
            Write-Host "    https://www.rust-lang.org/tools/install" -ForegroundColor Cyan
            exit 1
        }
        Write-Host "    ✓ Naga installed successfully!" -ForegroundColor Green
    }
} catch {
    Write-Host "    ✗ Error checking naga: $_" -ForegroundColor Red
    Write-Host "    Try installing manually: cargo install naga-cli" -ForegroundColor Yellow
    exit 1
}

# Test 2: Quick validation of original shader
$originalShader = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\lightFieldComposer.wgsl"

Write-Host "`n[2] Testing original shader validation..." -ForegroundColor Yellow
Write-Host "    File: $originalShader" -ForegroundColor Gray

if (Test-Path $originalShader) {
    $result = naga "$originalShader" --validate 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✓ Original shader is valid!" -ForegroundColor Green
    } else {
        Write-Host "    ✗ Validation errors found:" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
    }
} else {
    Write-Host "    ✗ Shader file not found!" -ForegroundColor Red
}

# Test 3: Quick validation of enhanced shader
$enhancedShader = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\lightFieldComposerEnhanced.wgsl"

Write-Host "`n[3] Testing enhanced shader validation..." -ForegroundColor Yellow
Write-Host "    File: $enhancedShader" -ForegroundColor Gray

if (Test-Path $enhancedShader) {
    $result = naga "$enhancedShader" --validate 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✓ Enhanced shader is valid!" -ForegroundColor Green
    } else {
        Write-Host "    ✗ Validation errors found:" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
    }
} else {
    Write-Host "    ✗ Shader file not found!" -ForegroundColor Red
}

Write-Host "`n==== TEST COMPLETE ====" -ForegroundColor Green
Write-Host "If all tests passed, you can run the full validation with:" -ForegroundColor White
Write-Host "  .\validate_shaders.ps1" -ForegroundColor Cyan
Write-Host "`nFor individual naga commands, use:" -ForegroundColor White
Write-Host '  naga "path\to\shader.wgsl" --validate' -ForegroundColor Gray
Write-Host '  naga "path\to\shader.wgsl" --output "output.spv"' -ForegroundColor Gray
Write-Host '  naga "path\to\shader.wgsl" --hlsl-out "output.hlsl"' -ForegroundColor Gray