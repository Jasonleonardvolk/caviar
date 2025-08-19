# Precise line-based fix for velocityField.wgsl
Write-Host "`n=== Precise Fix for velocityField.wgsl ===" -ForegroundColor Cyan

$velocityPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"

if (Test-Path $velocityPath) {
    $lines = Get-Content $velocityPath
    $newLines = @()
    
    # Process line by line
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($i -eq 281 -and $lines[$i] -match '@compute.*workgroup_size') {
            # We're at line 282 (0-indexed), which has @compute
            # Check if next lines have the function with storage param
            if ($lines[$i+1] -match 'fn\s+advect_particles' -and $lines[$i+2] -match '@group.*@binding.*particles') {
                Write-Host "Found the pattern at line $($i+2)" -ForegroundColor Green
                
                # Insert storage declaration before @compute
                $newLines += ""
                $newLines += "// Storage buffer for particle advection"
                $newLines += "@group(0) @binding(5) var<storage, read_write> particles: array<vec4<f32>>;"
                $newLines += ""
                
                # Add the @compute line
                $newLines += $lines[$i]
                
                # Add the function line without the storage parameter
                $newLines += "fn advect_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {"
                
                # Skip the original function declaration lines (283-284)
                $i += 2
            } else {
                $newLines += $lines[$i]
            }
        } else {
            $newLines += $lines[$i]
        }
    }
    
    # Save the fixed file
    Copy-Item $velocityPath "$velocityPath.backup_precise" -Force
    $newLines | Set-Content $velocityPath
    
    Write-Host "Applied precise fix to velocityField.wgsl" -ForegroundColor Green
    
    # Validate
    $result = & naga $velocityPath 2>&1 | Out-String
    if (-not ($result -match "error")) {
        Write-Host "✅ velocityField.wgsl is now valid!" -ForegroundColor Green
    } else {
        Write-Host "❌ Still has errors - manual fix needed" -ForegroundColor Red
        Write-Host "Run: .\manual_velocity_instructions.ps1" -ForegroundColor Yellow
    }
}
