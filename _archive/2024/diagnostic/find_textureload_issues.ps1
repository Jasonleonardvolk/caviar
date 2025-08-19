Write-Host "`n🔍 SEARCHING FOR ALL TEXTURELOAD CALLS" -ForegroundColor Cyan

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$shaders = @("lenticularInterlace.wgsl", "propagation.wgsl", "velocityField.wgsl")

foreach ($shader in $shaders) {
    $path = Join-Path $shaderDir $shader
    Write-Host "`n📄 $shader" -ForegroundColor Yellow
    
    $content = Get-Content $path
    $lineNum = 0
    
    foreach ($line in $content) {
        $lineNum++
        if ($line -match 'textureLoad\s*\([^)]+\)') {
            # Count commas to determine argument count
            $match = [regex]::Match($line, 'textureLoad\s*\(([^)]+)\)')
            $args = $match.Groups[1].Value
            $commaCount = ($args.ToCharArray() | Where-Object {$_ -eq ','}).Count
            $argCount = $commaCount + 1
            
            if ($argCount -eq 3) {
                # Check if it's a storage texture
                if ($line -match 'temp_buffer|output_field|frequency_domain|transfer_function|velocity_out|flow_vis_out|multiview_buffer|debug_') {
                    Write-Host "   Line $lineNum`: $($line.Trim())" -ForegroundColor Red
                    Write-Host "   ❌ texture_storage_2d with 3 args - needs fix!" -ForegroundColor Red
                } elseif ($line -match 'wavefield_texture|quilt_texture|previous_frame') {
                    Write-Host "   Line $lineNum`: $($line.Trim())" -ForegroundColor Green
                    Write-Host "   ✅ texture_2d with 3 args - correct!" -ForegroundColor Green
                } else {
                    Write-Host "   Line $lineNum`: $($line.Trim())" -ForegroundColor Yellow
                    Write-Host "   ⚠️  Unknown texture type - check manually" -ForegroundColor Yellow
                }
            } elseif ($argCount -eq 2) {
                Write-Host "   Line $lineNum`: $($line.Trim())" -ForegroundColor Green
                Write-Host "   ✅ 2 args - correct for texture_storage_2d" -ForegroundColor Green
            }
        }
    }
}

Write-Host "`n📋 TEXTURE TYPE REFERENCE:" -ForegroundColor Cyan
Write-Host "texture_storage_2d types (2 args only):" -ForegroundColor Yellow
Write-Host "  - temp_buffer, output_field, frequency_domain, transfer_function" -ForegroundColor White
Write-Host "  - velocity_out, flow_vis_out, multiview_buffer, debug_*" -ForegroundColor White
Write-Host "`ntexture_2d types (3 args with mip level):" -ForegroundColor Yellow
Write-Host "  - wavefield_texture, quilt_texture, previous_frame" -ForegroundColor White