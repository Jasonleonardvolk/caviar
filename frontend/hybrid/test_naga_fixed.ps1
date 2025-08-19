# test_naga_fixed.ps1
# Fixed version for naga 26.0.0 syntax

Write-Host "`n==== NAGA SHADER VALIDATOR TEST (v26.0.0) ====" -ForegroundColor Cyan
Write-Host "Testing naga installation and basic validation`n" -ForegroundColor White

# Test 1: Check if naga is installed
Write-Host "[1] Checking naga installation..." -ForegroundColor Yellow
try {
    $nagaVersion = naga --version 2>&1
    Write-Host "    ✓ Naga is installed: $nagaVersion" -ForegroundColor Green
    
    # Show help to understand correct syntax
    Write-Host "`n[2] Checking naga syntax..." -ForegroundColor Yellow
    naga --help 2>&1 | Select-String -Pattern "validate|wgsl" | ForEach-Object {
        Write-Host "    $_" -ForegroundColor Gray
    }
} catch {
    Write-Host "    ✗ Error checking naga: $_" -ForegroundColor Red
    exit 1
}

# Test different validation syntaxes
$originalShader = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\lightFieldComposer.wgsl"
$enhancedShader = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\lightFieldComposerEnhanced.wgsl"

Write-Host "`n[3] Testing validation syntaxes..." -ForegroundColor Yellow

# Try different command formats
Write-Host "`n    Testing format 1: naga validate <file>" -ForegroundColor Gray
$result1 = naga validate "$originalShader" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "    ✓ Format 1 works!" -ForegroundColor Green
} else {
    Write-Host "    ✗ Format 1 failed: $result1" -ForegroundColor Yellow
}

Write-Host "`n    Testing format 2: naga <file>" -ForegroundColor Gray
$result2 = naga "$originalShader" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "    ✓ Format 2 works!" -ForegroundColor Green
} else {
    Write-Host "    ✗ Format 2 failed: $result2" -ForegroundColor Yellow
}

Write-Host "`n    Testing format 3: naga info <file>" -ForegroundColor Gray
$result3 = naga info "$originalShader" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "    ✓ Format 3 works!" -ForegroundColor Green
    Write-Host $result3 -ForegroundColor Gray
} else {
    Write-Host "    ✗ Format 3 failed: $result3" -ForegroundColor Yellow
}

# Try to convert to SPIR-V (often the most reliable test)
Write-Host "`n[4] Testing SPIR-V conversion..." -ForegroundColor Yellow
$spirvOutput = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\test.spv"

# Try different output syntaxes
Write-Host "    Testing: naga ""$originalShader"" ""$spirvOutput""" -ForegroundColor Gray
naga "$originalShader" "$spirvOutput" 2>&1 | Out-Null

if (Test-Path $spirvOutput) {
    Write-Host "    ✓ SPIR-V conversion successful!" -ForegroundColor Green
    Remove-Item $spirvOutput
} else {
    Write-Host "    Trying: naga convert ""$originalShader"" --output ""$spirvOutput""" -ForegroundColor Gray
    naga convert "$originalShader" --output "$spirvOutput" 2>&1 | Out-Null
    
    if (Test-Path $spirvOutput) {
        Write-Host "    ✓ SPIR-V conversion successful with 'convert' command!" -ForegroundColor Green
        Remove-Item $spirvOutput
    } else {
        Write-Host "    ✗ Could not convert to SPIR-V" -ForegroundColor Red
    }
}

Write-Host "`n==== DIAGNOSTICS COMPLETE ====" -ForegroundColor Green
Write-Host "Based on the tests above, use the working command format." -ForegroundColor White
Write-Host "`nTo see all available commands:" -ForegroundColor White
Write-Host "  naga --help" -ForegroundColor Cyan