# Fix remaining shader issues
Write-Host "`n=== Fixing Remaining Shader Issues ===" -ForegroundColor Cyan

# 1. Fix propagation.wgsl (still JSON)
$propagationPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\propagation.wgsl"
if (Test-Path $propagationPath) {
    $content = Get-Content $propagationPath -Raw
    if ($content -match '^\s*{') {
        Write-Host "`nFixing propagation.wgsl (JSON format)..." -ForegroundColor Yellow
        
        # Extract the actual WGSL content from JSON
        if ($content -match '"content":\s*"([^"]+)"') {
            # It's wrapped in JSON, extract the content
            $jsonObj = $content | ConvertFrom-Json -ErrorAction SilentlyContinue
            if ($jsonObj -and $jsonObj.content) {
                # Unescape the content
                $wgslContent = $jsonObj.content -replace '\\n', "`n" -replace '\\r', "`r" -replace '\\"', '"' -replace '\\\\', '\'
                
                # Save backup
                Move-Item $propagationPath "$propagationPath.json_backup" -Force
                $wgslContent | Set-Content $propagationPath -NoNewline
                Write-Host "Extracted WGSL content from JSON wrapper" -ForegroundColor Green
            }
        } else {
            # Use the already fixed propagation.wgsl from frontend/shaders
            $sourcePath = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"
            if (Test-Path $sourcePath) {
                Copy-Item $sourcePath $propagationPath -Force
                Write-Host "Copied working propagation.wgsl from frontend/shaders" -ForegroundColor Green
            }
        }
    }
}

# 2. Fix lenticularInterlace.wgsl textureLoad arguments
$lenticularPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\lenticularInterlace.wgsl"
if (Test-Path $lenticularPath) {
    Write-Host "`nFixing lenticularInterlace.wgsl textureLoad calls..." -ForegroundColor Yellow
    $content = Get-Content $lenticularPath -Raw
    
    # Fix textureLoad with 3 arguments to 2 arguments (remove mip level)
    # textureLoad(texture, coord, 0) -> textureLoad(texture, coord)
    $content = $content -replace 'textureLoad\(([^,]+),\s*([^,]+),\s*0\)', 'textureLoad($1, $2)'
    
    $content | Set-Content $lenticularPath -NoNewline
    Write-Host "Fixed textureLoad argument count" -ForegroundColor Green
}

# 3. Fix velocityField.wgsl storage buffer parameters
$velocityPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"
if (Test-Path $velocityPath) {
    Write-Host "`nFixing velocityField.wgsl storage buffer parameters..." -ForegroundColor Yellow
    $content = Get-Content $velocityPath -Raw
    
    # Find the function with @group in parameters and fix it
    # This is around line 284 based on the error
    $pattern = '(fn\s+\w+\s*\([^@]*?)(@group\(\d+\)\s*@binding\(\d+\)\s*var<[^>]+>\s*\w+:\s*[^,)]+)(.*?\))'
    
    if ($content -match $pattern) {
        # Extract the storage buffer declaration
        $beforeParams = $matches[1]
        $storageDecl = $matches[2]
        $afterParams = $matches[3]
        
        # Convert to proper format
        $storageDecl = $storageDecl -replace ',\s*$', ''
        $moduleDecl = "`n// Storage buffer moved from function parameters`n$storageDecl;`n`n"
        
        # Insert before the function and remove from parameters
        $funcPattern = "((?:@compute[^\n]*\n)?)(fn\s+\w+\s*\([^)]*@group[^)]+\))"
        $content = $content -replace $funcPattern, {
            $compute = $matches[1]
            $func = $matches[2]
            
            # Remove the @group parameter from function
            $cleanFunc = $func -replace ',?\s*@group\(\d+\)\s*@binding\(\d+\)\s*var<[^>]+>\s*\w+:\s*[^,)]+', ''
            
            return $moduleDecl + $compute + $cleanFunc
        }
        
        $content | Set-Content $velocityPath -NoNewline
        Write-Host "Moved storage buffer declaration to module level" -ForegroundColor Green
    } else {
        Write-Host "Could not find pattern to fix - manual edit needed" -ForegroundColor Red
    }
}

Write-Host "`n=== Running final validation ===" -ForegroundColor Cyan
