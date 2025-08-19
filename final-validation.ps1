# Final Validation Check
Write-Host "`n🎯 FINAL VALIDATION CHECK" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

Write-Host "`n✅ Fix Applied:" -ForegroundColor Green
Write-Host "   applyPhaseLUT.wgsl - Changed textureSample to textureSampleLevel" -ForegroundColor White
Write-Host "   (Required for compute shaders)" -ForegroundColor Gray

Write-Host "`n🔍 Running validation..." -ForegroundColor Yellow

$output = node tools/shaders/validate_and_report.mjs --dir=frontend/lib/webgpu/shaders --limits=latest.ios --targets=naga --strict 2>&1 | Out-String

# Check for success
if ($output -match "50.*pass" -or $output -match "Passed:\s*50") {
    Write-Host "`n🎉 SUCCESS! 50/50 SHADERS PASS!" -ForegroundColor Green
    Write-Host "`n🚀 READY TO SHIP!" -ForegroundColor Magenta
    Write-Host "`nRun: .\tools\release\IrisOneButton.ps1" -ForegroundColor White
} else {
    # Parse results
    if ($output -match "Passed:\s*(\d+)") {
        $passed = $Matches[1]
        Write-Host "`n📊 Passed: $passed/50" -ForegroundColor Yellow
    }
    
    if ($output -match "Failed:\s*(\d+)") {
        $failed = $Matches[1]
        if ($failed -eq "0") {
            Write-Host "✅ No failures!" -ForegroundColor Green
        } else {
            Write-Host "❌ Failures: $failed" -ForegroundColor Red
        }
    }
}

Write-Host "`n=====================================`n" -ForegroundColor Cyan
