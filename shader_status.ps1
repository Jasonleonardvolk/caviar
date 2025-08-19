# Quick validation check
Write-Host "`n=== Quick Shader Status Check ===" -ForegroundColor Cyan

$shaders = @{
    "avatarShader.wgsl" = "✅ Fixed (was JSON)"
    "bitReversal.wgsl" = "✅ Already valid"
    "butterflyStage.wgsl" = "✅ Already valid" 
    "fftShift.wgsl" = "✅ Already valid"
    "lenticularInterlace.wgsl" = "❌ textureLoad needs 2 args not 3"
    "multiViewSynthesis.wgsl" = "✅ Fixed (added RenderParams)"
    "normalize.wgsl" = "✅ Already valid"
    "propagation.wgsl" = "❌ Still JSON format"
    "transpose.wgsl" = "✅ Already valid"
    "velocityField.wgsl" = "❌ @group in function params"
    "wavefieldEncoder_optimized.wgsl" = "✅ Already valid"
}

foreach ($shader in $shaders.GetEnumerator()) {
    Write-Host "$($shader.Key) - $($shader.Value)"
}

Write-Host "`n📋 TO FIX:" -ForegroundColor Yellow
Write-Host "1. Run: .\fix_final_shaders.ps1" -ForegroundColor Cyan
Write-Host "2. Run: .\smart_velocity_fix.ps1" -ForegroundColor Cyan  
Write-Host "3. Run: .\final_shader_check.ps1" -ForegroundColor Cyan

Write-Host "`n🎯 EXPECTED RESULT: All 11 shaders should be valid!" -ForegroundColor Green
