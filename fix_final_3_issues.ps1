# Fix the 3 remaining shader issues

Write-Host "`n=== Fixing 3 Remaining Shader Issues ===" -ForegroundColor Cyan

# 1. Fix lenticularInterlace.wgsl - textureLoad needs 3 args (add mip level 0)
Write-Host "`n1. Fixing lenticularInterlace.wgsl (line 367)..." -ForegroundColor Yellow
$lenticularPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\lenticularInterlace.wgsl"

if (Test-Path $lenticularPath) {
    $content = Get-Content $lenticularPath -Raw
    
    # This texture apparently needs 3 arguments - add mip level 0
    # Line 367: return textureLoad(quilt_texture, quilt_coord);
    $content = $content -replace 'return textureLoad\(quilt_texture, quilt_coord\);', 'return textureLoad(quilt_texture, quilt_coord, 0);'
    
    $content | Set-Content $lenticularPath -NoNewline
    Write-Host "Added mip level to textureLoad at line 367" -ForegroundColor Green
}

# 2. Fix propagation.wgsl in webgpu/shaders - line 476 has @group in function params
Write-Host "`n2. Fixing propagation.wgsl (line 476)..." -ForegroundColor Yellow
$propagationPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\propagation.wgsl"

if (Test-Path $propagationPath) {
    $lines = Get-Content $propagationPath
    $newLines = @()
    
    # Find and fix line 476 with @group in function parameters
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($i -eq 475 -and $lines[$i] -match '@group.*@binding.*multiview_buffer') {
            Write-Host "Found @group in function params at line $($i+1)" -ForegroundColor Yellow
            
            # Add storage declaration before the function
            $newLines += ""
            $newLines += "// Multiview buffer for synthesis"
            $newLines += "@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;"
            $newLines += ""
            
            # Skip the line with @group parameter
            continue
        } elseif ($i -eq 476 -and $lines[$i-1] -match '@group.*multiview_buffer') {
            # This is the closing ) { of the function, keep it but without the parameter
            $newLines += "                        @builtin(global_invocation_id) global_id: vec3<u32>) {"
        } else {
            $newLines += $lines[$i]
        }
    }
    
    $newLines | Set-Content $propagationPath
    Write-Host "Moved multiview_buffer storage declaration to module level" -ForegroundColor Green
}

# 3. Fix velocityField.wgsl - line 371 has @group in function params
Write-Host "`n3. Fixing velocityField.wgsl (line 371)..." -ForegroundColor Yellow
$velocityPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"

if (Test-Path $velocityPath) {
    $lines = Get-Content $velocityPath
    $newLines = @()
    $skipNext = $false
    
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($skipNext) {
            $skipNext = $false
            continue
        }
        
        # Check for @group around line 370-371
        if ($i -ge 369 -and $i -le 372 -and $lines[$i] -match '@group.*@binding') {
            Write-Host "Found @group at line $($i+1): $($lines[$i])" -ForegroundColor Yellow
            
            # Extract the storage declaration
            if ($lines[$i] -match '@group\((\d+)\)\s*@binding\((\d+)\)\s*var<([^>]+)>\s*(\w+):\s*(.+)') {
                $group = $matches[1]
                $binding = $matches[2]
                $varType = $matches[3]
                $varName = $matches[4]
                $arrayType = $matches[5]
                
                # Add module-level declaration
                $newLines += ""
                $newLines += "// Storage buffer for $varName"
                $newLines += "@group($group) @binding($binding) var<$varType> ${varName}: $arrayType;"
                $newLines += ""
                
                # Skip this line and potentially the next if it's a continuation
                if ($lines[$i+1] -match '^\s*\)') {
                    $skipNext = $true
                }
                continue
            }
        }
        
        $newLines += $lines[$i]
    }
    
    $newLines | Set-Content $velocityPath
    Write-Host "Fixed storage buffer declarations" -ForegroundColor Green
}

# Final validation
Write-Host "`n=== Final Validation ===" -ForegroundColor Cyan
$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"

Get-ChildItem "$shaderDir\*.wgsl" | ForEach-Object {
    $result = & naga $_.FullName 2>&1 | Out-String
    if (-not ($result -match "error")) {
        Write-Host "✅ $($_.Name)" -ForegroundColor Green
    } else {
        Write-Host "❌ $($_.Name)" -ForegroundColor Red
        $error = $result -split "`n" | Select-Object -First 3
        $error | ForEach-Object { Write-Host "   $_" -ForegroundColor DarkRed }
    }
}
