# Specific fix for velocityField.wgsl line 371 flow_vis_out

Write-Host "=== FIXING velocityField.wgsl flow_vis_out ===" -ForegroundColor Cyan

$file = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\velocityField.wgsl"

if (Test-Path $file) {
    $lines = Get-Content $file
    
    # Show the problem
    Write-Host "`nProblem at line 371:" -ForegroundColor Yellow
    Write-Host "370: $($lines[369])" -ForegroundColor DarkGray
    Write-Host "371: $($lines[370])" -ForegroundColor Red
    Write-Host "372: $($lines[371])" -ForegroundColor DarkGray
    
    # Fix: Move the @group line above the function
    $newLines = @()
    $fixed = $false
    
    for ($i = 0; $i -lt $lines.Count; $i++) {
        # When we hit line 370 (index 369) which has the function declaration
        if ($i -eq 369 -and $lines[$i] -match 'fn visualize_flow') {
            # Add the storage declaration before the function
            $newLines += ""
            $newLines += "// Storage texture for flow visualization"
            $newLines += "@group(0) @binding(6) var flow_vis_out: texture_storage_2d<rgba8unorm, write>;"
            $newLines += ""
            $newLines += "fn visualize_flow(@builtin(global_invocation_id) global_id: vec3<u32>) {"
            
            # Skip the next line which has the @group parameter
            $i++
            $fixed = $true
        } else {
            $newLines += $lines[$i]
        }
    }
    
    if ($fixed) {
        $newLines | Set-Content $file
        Write-Host "`nFixed! Moved flow_vis_out to module level" -ForegroundColor Green
        
        # Validate
        $result = & naga $file 2>&1 | Out-String
        if (-not ($result -match "error")) {
            Write-Host "✅ velocityField.wgsl is now valid!" -ForegroundColor Green
        } else {
            Write-Host "❌ Still has errors:" -ForegroundColor Red
            $result -split "`n" | Select-Object -First 5 | ForEach-Object { Write-Host $_ -ForegroundColor DarkRed }
        }
    } else {
        Write-Host "Could not find the pattern to fix" -ForegroundColor Red
    }
}
