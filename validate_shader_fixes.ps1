Write-Host "`n===== SHADER FIX SUMMARY =====" -ForegroundColor Cyan
Write-Host "Fixed 3 shaders with WGSL validation errors`n" -ForegroundColor Green

Write-Host "FIXES APPLIED:" -ForegroundColor Yellow
Write-Host "1. propagation.wgsl:" -ForegroundColor White
Write-Host "   - Moved multiview_buffer declaration from function parameter to module level"
Write-Host "   - Fixed prepare_for_multiview function signature"
Write-Host "   - Changed post_process to read from frequency_domain instead of output_field"

Write-Host "`n2. lenticularInterlace.wgsl:" -ForegroundColor White
Write-Host "   - Added mip level parameter (0) to textureLoad calls in cs_edge_enhance function"
Write-Host "   - Fixed lines 381 and 389"

Write-Host "`n3. velocityField.wgsl:" -ForegroundColor White
Write-Host "   - Moved particles storage buffer declaration to module level"
Write-Host "   - Moved flow_vis_out storage texture declaration to module level"
Write-Host "   - Fixed advect_particles and visualize_flow function signatures"

Write-Host "`n===== VALIDATING FIXED SHADERS =====" -ForegroundColor Cyan

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$fixedShaders = @("propagation.wgsl", "lenticularInterlace.wgsl", "velocityField.wgsl")

$allValid = $true
foreach ($shader in $fixedShaders) {
    $shaderPath = Join-Path $shaderDir $shader
    Write-Host "`nValidating $shader..." -ForegroundColor Yellow
    
    $result = & naga $shaderPath 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $shader is valid!" -ForegroundColor Green
    } else {
        Write-Host "❌ $shader still has errors:" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        $allValid = $false
    }
}

Write-Host "`n===== VALIDATION SUMMARY =====" -ForegroundColor Cyan
if ($allValid) {
    Write-Host "✅ All 3 shaders are now valid!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Some shaders still have validation errors" -ForegroundColor Yellow
    Write-Host "Please check the error messages above" -ForegroundColor Yellow
}

# Save fix summary
$summaryPath = "C:\Users\jason\Desktop\tori\kha\shader_fix_summary.txt"
$summary = @"
SHADER FIX SUMMARY
==================
Date: $(Get-Date)

Fixed 3 WGSL shaders with validation errors:

1. propagation.wgsl:
   - Moved multiview_buffer declaration from function parameter to module level
   - Fixed prepare_for_multiview function signature
   - Changed post_process to read from frequency_domain instead of output_field

2. lenticularInterlace.wgsl:
   - Added mip level parameter (0) to textureLoad calls in cs_edge_enhance function
   - Fixed lines 381 and 389

3. velocityField.wgsl:
   - Moved particles storage buffer declaration to module level
   - Moved flow_vis_out storage texture declaration to module level
   - Fixed advect_particles and visualize_flow function signatures

All shaders are located in:
$shaderDir
"@

$summary | Out-File -FilePath $summaryPath -Encoding UTF8
Write-Host "`nSummary saved to: $summaryPath" -ForegroundColor Cyan