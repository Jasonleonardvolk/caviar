# Comprehensive fix for all @group in function parameter issues
Write-Host "`n=== Comprehensive Fix for @group Parameters ===" -ForegroundColor Cyan

function Fix-GroupInFunctionParams {
    param($filePath)
    
    if (-not (Test-Path $filePath)) {
        Write-Host "File not found: $filePath" -ForegroundColor Red
        return
    }
    
    Write-Host "`nProcessing: $filePath" -ForegroundColor Yellow
    
    $content = Get-Content $filePath -Raw
    $originalContent = $content
    
    # Pattern to find functions with @group parameters
    # This matches compute functions with storage buffer parameters
    $pattern = '(@compute[^\n]*\n)(fn\s+\w+\s*\([^)]*?)(@group\(\d+\)\s*@binding\(\d+\)\s*var<[^>]+>\s*\w+:\s*[^),]+)(.*?\)\s*{)'
    
    while ($content -match $pattern) {
        $computeDirective = $matches[1]
        $funcStart = $matches[2]
        $storageParam = $matches[3]
        $funcEnd = $matches[4]
        
        # Extract storage declaration details
        if ($storageParam -match '@group\((\d+)\)\s*@binding\((\d+)\)\s*var<([^>]+)>\s*(\w+):\s*(.+)') {
            $storageDecl = "@group($($matches[1])) @binding($($matches[2])) var<$($matches[3])> $($matches[4]): $($matches[5]);"
            
            # Remove the storage parameter from function
            $cleanFunc = $funcStart -replace ',\s*$', ''
            $cleanEnd = $funcEnd -replace '^\s*,', ''
            
            # Insert storage declaration before function
            $replacement = "`n// Storage buffer moved from function parameters`n$storageDecl`n`n$computeDirective$cleanFunc$cleanEnd"
            
            # Replace in content
            $content = $content.Replace($matches[0], $replacement)
            
            Write-Host "Moved storage buffer to module level" -ForegroundColor Green
        }
    }
    
    # Also handle the multiview case from propagation.wgsl
    $multiviewPattern = 'fn\s+prepare_for_multiview\s*\([^)]*\n\s*@group\(\d+\)\s*@binding\(\d+\)\s*var\s+(\w+):\s*([^)]+)\)'
    if ($content -match $multiviewPattern) {
        $varName = $matches[1]
        $varType = $matches[2]
        
        # Extract @group and @binding numbers
        if ($content -match "@group\((\d+)\)\s*@binding\((\d+)\)\s*var\s+$varName") {
            $group = $matches[1]
            $binding = $matches[2]
            
            $storageDecl = "@group($group) @binding($binding) var $varName: $varType;"
            
            # Replace the function to remove the parameter
            $content = $content -replace "(fn\s+prepare_for_multiview\s*\()[^)]+(@group[^)]+)\)", '$1@builtin(global_invocation_id) global_id: vec3<u32>)'
            
            # Add the storage declaration before the function
            $content = $content -replace "(fn\s+prepare_for_multiview)", "$storageDecl`n`n`$1"
            
            Write-Host "Fixed prepare_for_multiview function" -ForegroundColor Green
        }
    }
    
    if ($content -ne $originalContent) {
        $content | Set-Content $filePath -NoNewline
        Write-Host "File updated successfully" -ForegroundColor Green
        return $true
    }
    
    return $false
}

# Fix all files
$files = @(
    "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl",
    "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\propagation.wgsl",
    "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"
)

foreach ($file in $files) {
    Fix-GroupInFunctionParams -filePath $file
}

# Fix lenticularInterlace.wgsl textureLoad issue
Write-Host "`nFixing lenticularInterlace.wgsl textureLoad..." -ForegroundColor Yellow
$lenticularPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\lenticularInterlace.wgsl"
if (Test-Path $lenticularPath) {
    $content = Get-Content $lenticularPath -Raw
    # Add mip level 0 as third parameter
    $content = $content -replace 'return textureLoad\(quilt_texture, quilt_coord\);', 'return textureLoad(quilt_texture, quilt_coord, 0);'
    $content | Set-Content $lenticularPath -NoNewline
    Write-Host "Added mip level parameter" -ForegroundColor Green
}

Write-Host "`n=== Validation ===" -ForegroundColor Cyan
