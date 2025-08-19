# validate_wgsl_simple.ps1
# Simple WGSL validation using naga 26.0.0

Write-Host "`n==== WGSL SHADER VALIDATION ====" -ForegroundColor Cyan
Write-Host "Using naga 26.0.0 for shader validation`n" -ForegroundColor White

$shaderPath = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl"
$originalShader = "$shaderPath\lightFieldComposer.wgsl"
$enhancedShader = "$shaderPath\lightFieldComposerEnhanced.wgsl"

function Test-Shader {
    param([string]$ShaderFile, [string]$Name)
    
    Write-Host "`n>>> Testing $Name" -ForegroundColor Green
    
    if (-not (Test-Path $ShaderFile)) {
        Write-Host "✗ File not found: $ShaderFile" -ForegroundColor Red
        return
    }
    
    $fileName = Split-Path $ShaderFile -Leaf
    $outputSpv = "$shaderPath\$([System.IO.Path]::GetFileNameWithoutExtension($fileName)).spv"
    
    Write-Host "Input:  $fileName" -ForegroundColor Gray
    Write-Host "Output: $([System.IO.Path]::GetFileName($outputSpv))" -ForegroundColor Gray
    
    # Method 1: Direct conversion (most reliable validation)
    Write-Host "`nAttempting SPIR-V conversion (validates WGSL)..." -ForegroundColor White
    
    # Try simple input/output syntax
    $output = naga "$ShaderFile" "$outputSpv" 2>&1
    $success = $LASTEXITCODE -eq 0
    
    if ($success -and (Test-Path $outputSpv)) {
        $size = (Get-Item $outputSpv).Length
        Write-Host "✓ Shader is valid! SPIR-V generated ($size bytes)" -ForegroundColor Green
        
        # Quick analysis of the shader
        $content = Get-Content $ShaderFile -Raw
        $lines = ($content -split "`n").Count
        $structs = ([regex]::Matches($content, "struct\s+\w+")).Count
        $functions = ([regex]::Matches($content, "fn\s+\w+")).Count
        $bindings = ([regex]::Matches($content, "@binding")).Count
        
        Write-Host "`nShader Statistics:" -ForegroundColor Cyan
        Write-Host "  Lines:     $lines" -ForegroundColor White
        Write-Host "  Structs:   $structs" -ForegroundColor White
        Write-Host "  Functions: $functions" -ForegroundColor White
        Write-Host "  Bindings:  $bindings" -ForegroundColor White
        
        # Clean up
        Remove-Item $outputSpv -Force
        return $true
    } else {
        Write-Host "✗ Validation failed!" -ForegroundColor Red
        if ($output) {
            Write-Host "Error details:" -ForegroundColor Yellow
            $output | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        }
        return $false
    }
}

# Validate both shaders
$originalValid = Test-Shader -ShaderFile $originalShader -Name "Original Light Field Composer"
$enhancedValid = Test-Shader -ShaderFile $enhancedShader -Name "Enhanced Light Field Composer"

Write-Host "`n==== VALIDATION SUMMARY ====" -ForegroundColor Cyan
if ($originalValid) {
    Write-Host "✓ Original shader: VALID" -ForegroundColor Green
} else {
    Write-Host "✗ Original shader: INVALID" -ForegroundColor Red
}

if ($enhancedValid) {
    Write-Host "✓ Enhanced shader: VALID" -ForegroundColor Green
} else {
    Write-Host "✗ Enhanced shader: INVALID" -ForegroundColor Red
}

Write-Host "`n==== ADDITIONAL NAGA COMMANDS ====" -ForegroundColor Yellow
Write-Host "To see naga help:" -ForegroundColor White
Write-Host "  naga --help" -ForegroundColor Gray
Write-Host "`nTo convert to SPIR-V:" -ForegroundColor White
Write-Host '  naga "input.wgsl" "output.spv"' -ForegroundColor Gray
Write-Host "`nTo convert to HLSL (if supported):" -ForegroundColor White
Write-Host '  naga "input.wgsl" "output.hlsl"' -ForegroundColor Gray
Write-Host "`nTo convert to Metal (if supported):" -ForegroundColor White
Write-Host '  naga "input.wgsl" "output.metal"' -ForegroundColor Gray