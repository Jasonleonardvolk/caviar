# Specific fix for velocityField.wgsl
Write-Host "`n=== Fixing velocityField.wgsl ===" -ForegroundColor Cyan

$velocityPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"

if (Test-Path $velocityPath) {
    # Read the file
    $lines = Get-Content $velocityPath
    
    # Find line 284 and surrounding context
    Write-Host "`nFound problematic function around line 284:" -ForegroundColor Yellow
    for ($i = 280; $i -lt 290 -and $i -lt $lines.Count; $i++) {
        if ($i -eq 283) {
            Write-Host "$($i+1): $($lines[$i])" -ForegroundColor Red
        } else {
            Write-Host "$($i+1): $($lines[$i])" -ForegroundColor DarkGray
        }
    }
    
    # Create fixed version
    $fixedLines = @()
    $storageBufferFound = $false
    $storageDeclaration = ""
    
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        
        # Check if this is the problematic line with @group in function parameters
        if ($line -match '@group.*@binding.*var.*storage.*particles' -and $line -match 'fn|^\s*@') {
            Write-Host "`nFound storage buffer in function parameters at line $($i+1)" -ForegroundColor Yellow
            
            # Extract the storage buffer declaration
            if ($line -match '(@group\(\d+\)\s*@binding\(\d+\)\s*var<[^>]+>\s*particles:\s*[^)]+)') {
                $storageDeclaration = $matches[1] + ";"
                $storageBufferFound = $true
                
                # Remove the storage buffer from the function parameters
                $line = $line -replace ',?\s*@group\(\d+\)\s*@binding\(\d+\)\s*var<[^>]+>\s*particles:\s*[^)]+', ''
            }
        }
        
        # If we found a function declaration after finding storage buffer, insert the declaration before it
        if ($storageBufferFound -and $line -match '^\s*fn\s+') {
            $fixedLines += ""
            $fixedLines += "// Storage buffer moved from function parameters"
            $fixedLines += $storageDeclaration
            $fixedLines += ""
            $storageBufferFound = $false
        }
        
        $fixedLines += $line
    }
    
    # Save the fixed file
    $fixedLines | Set-Content $velocityPath
    Write-Host "`nSaved fixed velocityField.wgsl" -ForegroundColor Green
}
