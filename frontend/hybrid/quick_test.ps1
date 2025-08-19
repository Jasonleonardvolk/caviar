# quick_test.ps1
# Quick test with correct naga 26.0.0 syntax

Write-Host "`nQuick naga validation test:" -ForegroundColor Cyan

$shader = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\lightFieldComposer.wgsl"

# Test 1: Validation only (no output file)
Write-Host "`nTest 1: Validation only" -ForegroundColor Yellow
Write-Host "Command: naga ""$shader""" -ForegroundColor Gray
naga "$shader"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Validation passed!" -ForegroundColor Green
} else {
    Write-Host "✗ Validation failed" -ForegroundColor Red
}

# Test 2: Convert to SPIR-V (validates + converts)
Write-Host "`nTest 2: Convert to SPIR-V" -ForegroundColor Yellow
$output = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\test.spv"
Write-Host "Command: naga ""$shader"" ""$output""" -ForegroundColor Gray
naga "$shader" "$output"
if ($LASTEXITCODE -eq 0 -and (Test-Path $output)) {
    $size = (Get-Item $output).Length
    Write-Host "✓ Conversion successful! Output: test.spv ($size bytes)" -ForegroundColor Green
    Remove-Item $output
} else {
    Write-Host "✗ Conversion failed" -ForegroundColor Red
}

Write-Host "`nNow run the full validation:" -ForegroundColor White
Write-Host "  .\validate_shaders_v26.ps1" -ForegroundColor Cyan