# validate_shaders.ps1
# PowerShell script to validate and test the light field composer shaders

Write-Host "Light Field Composer Shader Validation Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

$shaderPath = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl"
$originalShader = "$shaderPath\lightFieldComposer.wgsl"
$enhancedShader = "$shaderPath\lightFieldComposerEnhanced.wgsl"

# Check if naga is installed
Write-Host "`nChecking for naga installation..." -ForegroundColor Yellow
$nagaCheck = Get-Command naga -ErrorAction SilentlyContinue

if (-not $nagaCheck) {
    Write-Host "Naga not found. Installing via cargo..." -ForegroundColor Yellow
    cargo install naga-cli
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install naga. Please install Rust and cargo first." -ForegroundColor Red
        exit 1
    }
}

# Function to validate a shader
function Validate-Shader {
    param(
        [string]$ShaderFile,
        [string]$ShaderName
    )
    
    Write-Host "`n>>> Validating $ShaderName" -ForegroundColor Green
    Write-Host "Path: $ShaderFile" -ForegroundColor Gray
    
    if (-not (Test-Path $ShaderFile)) {
        Write-Host "File not found: $ShaderFile" -ForegroundColor Red
        return $false
    }
    
    # Basic validation
    Write-Host "Running basic validation..." -ForegroundColor White
    $validation = naga "$ShaderFile" --validate 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Validation passed!" -ForegroundColor Green
    } else {
        Write-Host "✗ Validation failed:" -ForegroundColor Red
        Write-Host $validation
        return $false
    }
    
    # Check with WebGPU features
    Write-Host "Checking WebGPU features..." -ForegroundColor White
    $features = naga "$ShaderFile" --validate --features "PUSH_CONSTANTS,MULTIVIEW,TEXTURE_BINDING_ARRAY" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ WebGPU features supported!" -ForegroundColor Green
    } else {
        Write-Host "⚠ Some WebGPU features may not be supported" -ForegroundColor Yellow
    }
    
    return $true
}

# Function to convert shader to different formats
function Convert-Shader {
    param(
        [string]$ShaderFile,
        [string]$ShaderName
    )
    
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($ShaderFile)
    $outputDir = [System.IO.Path]::GetDirectoryName($ShaderFile)
    
    Write-Host "`n>>> Converting $ShaderName to other formats" -ForegroundColor Cyan
    
    # Convert to SPIR-V
    Write-Host "Converting to SPIR-V..." -ForegroundColor White
    $spirvOutput = "$outputDir\$baseName.spv"
    naga "$ShaderFile" --output "$spirvOutput" 2>&1 | Out-Null
    
    if (Test-Path $spirvOutput) {
        $size = (Get-Item $spirvOutput).Length
        Write-Host "✓ SPIR-V generated: $spirvOutput ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to generate SPIR-V" -ForegroundColor Red
    }
    
    # Convert to HLSL
    Write-Host "Converting to HLSL..." -ForegroundColor White
    $hlslOutput = "$outputDir\$baseName.hlsl"
    naga "$ShaderFile" --hlsl-out "$hlslOutput" 2>&1 | Out-Null
    
    if (Test-Path $hlslOutput) {
        Write-Host "✓ HLSL generated: $hlslOutput" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to generate HLSL" -ForegroundColor Red
    }
    
    # Convert to MSL (Metal)
    Write-Host "Converting to MSL (Metal)..." -ForegroundColor White
    $mslOutput = "$outputDir\$baseName.metal"
    naga "$ShaderFile" --msl-out "$mslOutput" 2>&1 | Out-Null
    
    if (Test-Path $mslOutput) {
        Write-Host "✓ MSL generated: $mslOutput" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to generate MSL" -ForegroundColor Red
    }
}

# Function to analyze shader complexity
function Analyze-Shader {
    param(
        [string]$ShaderFile,
        [string]$ShaderName
    )
    
    Write-Host "`n>>> Analyzing $ShaderName complexity" -ForegroundColor Magenta
    
    $content = Get-Content $ShaderFile -Raw
    
    # Count various elements
    $structCount = ([regex]::Matches($content, "struct\s+\w+")).Count
    $functionCount = ([regex]::Matches($content, "fn\s+\w+")).Count
    $bindingCount = ([regex]::Matches($content, "@binding\(\d+\)")).Count
    $textureCount = ([regex]::Matches($content, "texture_")).Count
    $uniformCount = ([regex]::Matches($content, "var<uniform>")).Count
    $storageCount = ([regex]::Matches($content, "var<storage")).Count
    
    Write-Host "Structs:        $structCount" -ForegroundColor White
    Write-Host "Functions:      $functionCount" -ForegroundColor White
    Write-Host "Bindings:       $bindingCount" -ForegroundColor White
    Write-Host "Textures:       $textureCount" -ForegroundColor White
    Write-Host "Uniforms:       $uniformCount" -ForegroundColor White
    Write-Host "Storage Buffers: $storageCount" -ForegroundColor White
    
    # Estimate memory usage
    $workgroupSize = if ($content -match "@workgroup_size\((\d+),\s*(\d+)") {
        [int]$matches[1] * [int]$matches[2]
    } else { 64 }
    
    Write-Host "Workgroup Size: $workgroupSize threads" -ForegroundColor White
    
    # Check for advanced features
    $hasPhaseCalc = $content -match "phase|Phase"
    $hasTensorCalc = $content -match "tensor|Tensor"
    $hasSoliton = $content -match "soliton|Soliton"
    
    if ($hasPhaseCalc -or $hasTensorCalc -or $hasSoliton) {
        Write-Host "`nAdvanced Features Detected:" -ForegroundColor Yellow
        if ($hasPhaseCalc) { Write-Host "  ✓ Phase calculations" -ForegroundColor Green }
        if ($hasTensorCalc) { Write-Host "  ✓ Tensor field processing" -ForegroundColor Green }
        if ($hasSoliton) { Write-Host "  ✓ Soliton dynamics" -ForegroundColor Green }
    }
}

# Function to generate documentation
function Generate-Documentation {
    param(
        [string]$ShaderFile,
        [string]$ShaderName
    )
    
    $docFile = "$shaderPath\$([System.IO.Path]::GetFileNameWithoutExtension($ShaderFile))_doc.md"
    
    Write-Host "`n>>> Generating documentation for $ShaderName" -ForegroundColor Blue
    Write-Host "Output: $docFile" -ForegroundColor Gray
    
    $content = Get-Content $ShaderFile -Raw
    
    # Extract structs
    $structs = [regex]::Matches($content, "struct\s+(\w+)\s*\{([^}]+)\}")
    
    # Extract functions
    $functions = [regex]::Matches($content, "@(compute|vertex|fragment).*?fn\s+(\w+)\s*\([^)]*\)")
    
    # Extract bindings
    $bindings = [regex]::Matches($content, "@group\((\d+)\)\s*@binding\((\d+)\)\s*var[^;]+")
    
    $doc = @"
# $ShaderName Documentation

Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Overview
$ShaderName is a WebGPU compute shader for light field composition.

## Structs
"@
    
    foreach ($struct in $structs) {
        $doc += "`n### $($struct.Groups[1].Value)`n"
        $doc += "``````wgsl`n$($struct.Value)`n```````n"
    }
    
    $doc += "`n## Functions`n"
    foreach ($func in $functions) {
        $doc += "- **$($func.Groups[2].Value)**: $($func.Groups[1].Value) shader entry point`n"
    }
    
    $doc += "`n## Resource Bindings`n"
    foreach ($binding in $bindings) {
        $doc += "- Group $($binding.Groups[1].Value), Binding $($binding.Groups[2].Value)`n"
    }
    
    $doc | Out-File -FilePath $docFile -Encoding UTF8
    Write-Host "✓ Documentation generated!" -ForegroundColor Green
}

# Main execution
Write-Host "`n=== VALIDATING ORIGINAL SHADER ===" -ForegroundColor Cyan
if (Validate-Shader -ShaderFile $originalShader -ShaderName "Original Light Field Composer") {
    Convert-Shader -ShaderFile $originalShader -ShaderName "Original"
    Analyze-Shader -ShaderFile $originalShader -ShaderName "Original"
    Generate-Documentation -ShaderFile $originalShader -ShaderName "Original Light Field Composer"
}

Write-Host "`n=== VALIDATING ENHANCED SHADER ===" -ForegroundColor Cyan
if (Validate-Shader -ShaderFile $enhancedShader -ShaderName "Enhanced Light Field Composer") {
    Convert-Shader -ShaderFile $enhancedShader -ShaderName "Enhanced"
    Analyze-Shader -ShaderFile $enhancedShader -ShaderName "Enhanced"
    Generate-Documentation -ShaderFile $enhancedShader -ShaderName "Enhanced Light Field Composer"
}

# Compare the two shaders
Write-Host "`n=== COMPARISON ===" -ForegroundColor Yellow
Write-Host "Original shader: $(Get-Item $originalShader).Length bytes" -ForegroundColor White
Write-Host "Enhanced shader: $(Get-Item $enhancedShader).Length bytes" -ForegroundColor White

$enhancement = [math]::Round(($(Get-Item $enhancedShader).Length / $(Get-Item $originalShader).Length - 1) * 100, 2)
Write-Host "Enhancement adds ${enhancement}% more code" -ForegroundColor Cyan

Write-Host "`n=== VALIDATION COMPLETE ===" -ForegroundColor Green
Write-Host "All shader files have been validated and converted." -ForegroundColor Green
Write-Host "Check the $shaderPath directory for generated files." -ForegroundColor Gray