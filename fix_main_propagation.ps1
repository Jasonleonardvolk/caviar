# Fix the main propagation.wgsl file
Write-Host "=== Fixing main propagation.wgsl ===" -ForegroundColor Cyan

$file = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"

if (Test-Path $file) {
    # Read the file
    $lines = Get-Content $file
    
    # Show the problem area
    Write-Host "`nProblem at line 476:" -ForegroundColor Yellow
    for ($i = 473; $i -lt 478 -and $i -lt $lines.Count; $i++) {
        if ($i -eq 475) {
            Write-Host "$($i+1): $($lines[$i])" -ForegroundColor Red
        } else {
            Write-Host "$($i+1): $($lines[$i])" -ForegroundColor DarkGray
        }
    }
    
    # Fix: Move @group declaration to module level
    $newLines = @()
    $skipLine = -1
    
    for ($i = 0; $i -lt $lines.Count; $i++) {
        # Skip the line if marked
        if ($i -eq $skipLine) {
            continue
        }
        
        # Look for the prepare_for_multiview function
        if ($i -ge 473 -and $i -le 477) {
            if ($lines[$i] -match 'fn prepare_for_multiview') {
                # Found the function, insert storage declaration before it
                $newLines += ""
                $newLines += "// Multiview buffer for synthesis"
                $newLines += "@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;"
                $newLines += ""
                
                # Add the function line
                $newLines += $lines[$i]
                
                # Check if next line has the @group parameter
                if ($i+1 -lt $lines.Count -and $lines[$i+1] -match '@group.*@binding.*multiview_buffer') {
                    # Skip the @group line and fix the closing
                    $skipLine = $i + 1
                    # Add just the closing of function signature
                    $newLines += "                        @builtin(global_invocation_id) global_id: vec3<u32>) {"
                    $i++
                }
            } else {
                $newLines += $lines[$i]
            }
        } else {
            $newLines += $lines[$i]
        }
    }
    
    # Save the fixed file
    $newLines | Set-Content $file
    Write-Host "`nFixed! Moved multiview_buffer to module level" -ForegroundColor Green
    
    # Validate
    Write-Host "`nValidating..." -ForegroundColor Yellow
    $result = & naga $file 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ propagation.wgsl is now valid!" -ForegroundColor Green
    } else {
        Write-Host "❌ Still has errors:" -ForegroundColor Red
        $result | Select-Object -First 10
    }
}
