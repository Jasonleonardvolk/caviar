# Final Shader Gate Fix
Write-Host "`n🎯 FINAL SHADER FIX - Resolving applyPhaseLUT" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

Write-Host "`n✅ Applied Fixes:" -ForegroundColor Green
Write-Host ""
Write-Host "1. applyPhaseLUT.wgsl:" -ForegroundColor Yellow
Write-Host "   - Changed from storage buffer to texture/sampler for LUT" -ForegroundColor White
Write-Host "   - Unique bindings: 0,1,2 (buffers), 3,4 (texture/sampler)" -ForegroundColor White
Write-Host ""
Write-Host "2. phaseLUT.ts pipeline:" -ForegroundColor Yellow
Write-Host "   - Created matching pipeline with correct bind group" -ForegroundColor White
Write-Host "   - Matches the shader's binding layout exactly" -ForegroundColor White

# Test the fix
Write-Host "`n🔍 Testing shader validation..." -ForegroundColor Yellow

# Try different validator scripts
$validators = @(
    "tools\shaders\shader_quality_gate_v2.mjs",
    "tools\shaders\validate-wgsl.js",
    "tools\shaders\validate_and_report.mjs"
)

$validator = $null
foreach ($v in $validators) {
    if (Test-Path $v) {
        $validator = $v
        break
    }
}

if ($validator) {
    Write-Host "Using validator: $validator" -ForegroundColor Gray
    
    # Run with latest limits
    $output = & node $validator --dir=frontend --limits=latest --strict 2>&1 | Out-String
    
    # Check for success
    if ($output -match "0\s+fail" -or $output -match "pass.*50/50") {
        Write-Host "`n🎉 SUCCESS! All shaders pass validation!" -ForegroundColor Green
        Write-Host "`n✅ Ready to ship!" -ForegroundColor Magenta
    } else {
        # Parse results
        if ($output -match "(\d+)\s+fail") {
            $failures = $Matches[1]
            Write-Host "`n📊 Remaining failures: $failures" -ForegroundColor Yellow
        }
        
        if ($output -match "(\d+)\s+warn") {
            $warnings = $Matches[1]
            Write-Host "📊 Warnings: $warnings (non-blocking)" -ForegroundColor Gray
        }
        
        # Check specific files
        if ($output -match "applyPhaseLUT") {
            Write-Host "`n⚠️ applyPhaseLUT still flagged - check:" -ForegroundColor Yellow
            Write-Host "   1. Clear build caches" -ForegroundColor White
            Write-Host "   2. Ensure pipeline code uses the texture version" -ForegroundColor White
        }
    }
} else {
    Write-Host "⚠️ No validator found - run manually:" -ForegroundColor Yellow
    Write-Host "   npm run shaders:gate:latest" -ForegroundColor White
}

Write-Host "`n🚀 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Run: npm run shaders:gate:latest" -ForegroundColor White
Write-Host "2. If green, ship with: .\tools\release\IrisOneButton.ps1" -ForegroundColor White
Write-Host "3. If issues, check the validator output above" -ForegroundColor White
