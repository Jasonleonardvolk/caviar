# test_fixed_shaders.ps1
# Test the corrected WGSL shaders

Write-Host "`n==== TESTING FIXED WGSL SHADERS ====" -ForegroundColor Cyan
Write-Host "Verifying syntax corrections`n" -ForegroundColor White

$originalShader = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\lightFieldComposer.wgsl"
$enhancedShader = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\lightFieldComposerEnhanced.wgsl"

function Test-Shader {
    param([string]$Path, [string]$Name)
    
    Write-Host "`n>>> Testing $Name" -ForegroundColor Green
    Write-Host "Path: $Path" -ForegroundColor Gray
    
    # Validation test
    Write-Host "`nValidating..." -ForegroundColor Yellow
    $result = naga "$Path" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Validation PASSED!" -ForegroundColor Green
        
        # Try SPIR-V conversion as additional test
        $spvPath = "$Path.spv"
        Write-Host "Converting to SPIR-V..." -ForegroundColor Yellow
        naga "$Path" "$spvPath" 2>&1 | Out-Null
        
        if (Test-Path $spvPath) {
            $size = (Get-Item $spvPath).Length
            Write-Host "✓ SPIR-V generated successfully ($size bytes)" -ForegroundColor Green
            Remove-Item $spvPath
        }
        
        return $true
    } else {
        Write-Host "✗ Validation FAILED:" -ForegroundColor Red
        $result | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        return $false
    }
}

# Test both shaders
$originalOk = Test-Shader -Path $originalShader -Name "Original Light Field Composer"
$enhancedOk = Test-Shader -Path $enhancedShader -Name "Enhanced Light Field Composer"

Write-Host "`n==== RESULTS ====" -ForegroundColor Cyan
if ($originalOk -and $enhancedOk) {
    Write-Host "✓ Both shaders are now valid!" -ForegroundColor Green
    Write-Host "`nYou can now run the full validation suite:" -ForegroundColor White
    Write-Host "  .\validate_shaders_v26.ps1" -ForegroundColor Cyan
} else {
    Write-Host "✗ Some shaders still have errors" -ForegroundColor Red
    Write-Host "Please check the error messages above" -ForegroundColor Yellow
}
