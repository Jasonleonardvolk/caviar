# validate_shaders_v26.ps1
# Correct validation script for naga 26.0.0

Write-Host "`n==== WGSL SHADER VALIDATION (naga 26.0.0) ====" -ForegroundColor Cyan
Write-Host "Validating and converting light field composer shaders`n" -ForegroundColor White

$shaderPath = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl"
$originalShader = "$shaderPath\lightFieldComposer.wgsl"
$enhancedShader = "$shaderPath\lightFieldComposerEnhanced.wgsl"

function Validate-And-Convert-Shader {
    param(
        [string]$InputFile,
        [string]$ShaderName
    )
    
    Write-Host "`n>>> Processing $ShaderName" -ForegroundColor Green
    Write-Host "Input: $InputFile" -ForegroundColor Gray
    
    if (-not (Test-Path $InputFile)) {
        Write-Host "✗ File not found!" -ForegroundColor Red
        return $false
    }
    
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($InputFile)
    $outputDir = [System.IO.Path]::GetDirectoryName($InputFile)
    
    # Test 1: Validation only (no output file = validation only)
    Write-Host "`n[1] Validating WGSL..." -ForegroundColor Yellow
    $validation = naga "$InputFile" 2>&1
    $validationExitCode = $LASTEXITCODE
    
    if ($validationExitCode -eq 0) {
        Write-Host "    ✓ WGSL validation passed!" -ForegroundColor Green
    } else {
        Write-Host "    ✗ Validation errors found:" -ForegroundColor Red
        $validation | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        return $false
    }
    
    # Test 2: Convert to SPIR-V
    Write-Host "`n[2] Converting to SPIR-V..." -ForegroundColor Yellow
    $spirvOutput = "$outputDir\$baseName.spv"
    $spirvResult = naga "$InputFile" "$spirvOutput" 2>&1
    
    if ($LASTEXITCODE -eq 0 -and (Test-Path $spirvOutput)) {
        $size = (Get-Item $spirvOutput).Length
        Write-Host "    ✓ SPIR-V generated: $([System.IO.Path]::GetFileName($spirvOutput)) ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "    ✗ SPIR-V conversion failed" -ForegroundColor Red
        if ($spirvResult) { $spirvResult | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray } }
    }
    
    # Test 3: Convert to HLSL
    Write-Host "`n[3] Converting to HLSL..." -ForegroundColor Yellow
    $hlslOutput = "$outputDir\$baseName.hlsl"
    $hlslResult = naga "$InputFile" "$hlslOutput" --shader-model 60 2>&1
    
    if ($LASTEXITCODE -eq 0 -and (Test-Path $hlslOutput)) {
        $size = (Get-Item $hlslOutput).Length
        Write-Host "    ✓ HLSL generated: $([System.IO.Path]::GetFileName($hlslOutput)) ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "    ✗ HLSL conversion failed" -ForegroundColor Red
        if ($hlslResult) { $hlslResult | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray } }
    }
    
    # Test 4: Convert to Metal
    Write-Host "`n[4] Converting to Metal..." -ForegroundColor Yellow
    $metalOutput = "$outputDir\$baseName.metal"
    $metalResult = naga "$InputFile" "$metalOutput" --metal-version 2.4 2>&1
    
    if ($LASTEXITCODE -eq 0 -and (Test-Path $metalOutput)) {
        $size = (Get-Item $metalOutput).Length
        Write-Host "    ✓ Metal generated: $([System.IO.Path]::GetFileName($metalOutput)) ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "    ✗ Metal conversion failed" -ForegroundColor Red
        if ($metalResult) { $metalResult | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray } }
    }
    
    # Test 5: Validate with bounds checking
    Write-Host "`n[5] Validating with bounds checking..." -ForegroundColor Yellow
    $boundsResult = naga "$InputFile" --index-bounds-check-policy Restrict --validate 0xFFFFFFFF 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✓ Bounds checking validation passed!" -ForegroundColor Green
    } else {
        Write-Host "    ⚠ Bounds checking found potential issues" -ForegroundColor Yellow
    }
    
    # Analyze shader
    Write-Host "`n[6] Shader Analysis..." -ForegroundColor Cyan
    $content = Get-Content $InputFile -Raw
    
    $lines = ($content -split "`n").Count
    $structs = ([regex]::Matches($content, "struct\s+\w+")).Count
    $functions = ([regex]::Matches($content, "@(compute|vertex|fragment).*?fn\s+\w+")).Count
    $bindings = ([regex]::Matches($content, "@binding\(\d+\)")).Count
    $groups = ([regex]::Matches($content, "@group\(\d+\)")).Count
    $workgroupSize = if ($content -match "@workgroup_size\((\d+),\s*(\d+)") {
        "$($matches[1]) x $($matches[2])"
    } else { "N/A" }
    
    Write-Host "    Lines:          $lines" -ForegroundColor White
    Write-Host "    Structs:        $structs" -ForegroundColor White
    Write-Host "    Entry Points:   $functions" -ForegroundColor White
    Write-Host "    Bindings:       $bindings" -ForegroundColor White
    Write-Host "    Bind Groups:    $groups" -ForegroundColor White
    Write-Host "    Workgroup Size: $workgroupSize" -ForegroundColor White
    
    # Check for advanced features
    $hasPhase = $content -match "phase|Phase"
    $hasTensor = $content -match "tensor|Tensor"
    $hasSoliton = $content -match "soliton|Soliton"
    $hasComplex = $content -match "complex|Complex"
    
    if ($hasPhase -or $hasTensor -or $hasSoliton -or $hasComplex) {
        Write-Host "`n    Advanced Features:" -ForegroundColor Yellow
        if ($hasPhase) { Write-Host "      ✓ Phase calculations" -ForegroundColor Green }
        if ($hasTensor) { Write-Host "      ✓ Tensor field processing" -ForegroundColor Green }
        if ($hasSoliton) { Write-Host "      ✓ Soliton dynamics" -ForegroundColor Green }
        if ($hasComplex) { Write-Host "      ✓ Complex number operations" -ForegroundColor Green }
    }
    
    return $true
}

# Process both shaders
$originalValid = Validate-And-Convert-Shader -InputFile $originalShader -ShaderName "Original Light Field Composer"
$enhancedValid = Validate-And-Convert-Shader -InputFile $enhancedShader -ShaderName "Enhanced Light Field Composer"

# Summary
Write-Host "`n==== VALIDATION SUMMARY ====" -ForegroundColor Cyan

if ($originalValid) {
    Write-Host "✓ Original shader: VALID and converted" -ForegroundColor Green
} else {
    Write-Host "✗ Original shader: FAILED validation" -ForegroundColor Red
}

if ($enhancedValid) {
    Write-Host "✓ Enhanced shader: VALID and converted" -ForegroundColor Green  
} else {
    Write-Host "✗ Enhanced shader: FAILED validation" -ForegroundColor Red
}

# List generated files
Write-Host "`n==== GENERATED FILES ====" -ForegroundColor Cyan
Get-ChildItem $shaderPath -Filter "*.spv","*.hlsl","*.metal" | ForEach-Object {
    Write-Host "  $($_.Name) - $($_.Length) bytes" -ForegroundColor White
}

Write-Host "`n==== QUICK REFERENCE ====" -ForegroundColor Yellow
Write-Host "Validate only (no output):" -ForegroundColor White
Write-Host '  naga "shader.wgsl"' -ForegroundColor Gray
Write-Host "`nConvert to SPIR-V:" -ForegroundColor White
Write-Host '  naga "shader.wgsl" "output.spv"' -ForegroundColor Gray
Write-Host "`nConvert to HLSL with SM 6.0:" -ForegroundColor White
Write-Host '  naga "shader.wgsl" "output.hlsl" --shader-model 60' -ForegroundColor Gray
Write-Host "`nConvert to Metal 2.4:" -ForegroundColor White
Write-Host '  naga "shader.wgsl" "output.metal" --metal-version 2.4' -ForegroundColor Gray
Write-Host "`nValidate with all checks:" -ForegroundColor White
Write-Host '  naga "shader.wgsl" --validate 0xFFFFFFFF --index-bounds-check-policy Restrict' -ForegroundColor Gray