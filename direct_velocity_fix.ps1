# Manual fix for velocityField.wgsl - Direct edit
Write-Host "`n=== Direct Fix for velocityField.wgsl ===" -ForegroundColor Cyan

$velocityPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"

if (Test-Path $velocityPath) {
    $lines = Get-Content $velocityPath
    $newLines = @()
    $i = 0
    
    while ($i -lt $lines.Count) {
        # Look for line 283-284 pattern
        if ($i -eq 283 -and $lines[$i] -match '@group.*@binding.*particles') {
            Write-Host "Found storage buffer in function params at line $($i+1)" -ForegroundColor Yellow
            
            # Add the storage buffer declaration before the function
            $newLines += ""
            $newLines += "// Storage buffer for particle advection"
            $newLines += "@group(0) @binding(5) var<storage, read_write> particles: array<vec4<f32>>;"
            $newLines += ""
            
            # Add line 282 (@compute)
            $newLines += $lines[$i-1]
            
            # Fix line 283 - remove the storage parameter
            $funcLine = $lines[$i-2]
            $funcLine = $funcLine -replace ',?\s*$', ') {'
            $newLines += $funcLine
            
            # Skip the original lines 283-284
            $i += 2
        } else {
            $newLines += $lines[$i]
            $i++
        }
    }
    
    # Save the fixed version
    $newLines | Set-Content $velocityPath
    Write-Host "Fixed velocityField.wgsl" -ForegroundColor Green
}
