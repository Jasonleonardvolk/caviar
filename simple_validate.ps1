Write-Host "`n🎯 FINAL WGSL VALIDATION" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$shaders = @(
    "avatarShader.wgsl",
    "bitReversal.wgsl",
    "butterflyStage.wgsl",
    "fftShift.wgsl",
    "lenticularInterlace.wgsl",
    "multiViewSynthesis.wgsl",
    "normalize.wgsl",
    "propagation.wgsl",
    "transpose.wgsl",
    "velocityField.wgsl",
    "wavefieldEncoder_optimized.wgsl"
)

$validCount = 0
$totalCount = $shaders.Count

foreach ($shader in $shaders) {
    $path = Join-Path $shaderDir $shader
    Write-Host "▶ Checking $shader ..." -NoNewline
    
    $result = & naga $path 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅" -ForegroundColor Green
        $validCount++
    }
    else {
        Write-Host " ❌" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
    }
}

Write-Host "`n📊 RESULTS: $validCount / $totalCount shaders valid" -ForegroundColor Cyan

if ($validCount -eq $totalCount) {
    Write-Host "`n🎉 ALL SHADERS ARE VALID! 🎉" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. npx tsx scripts/bundleShaders.ts" -ForegroundColor White
    Write-Host "2. .\START_TORI_HARDENED.bat -Force" -ForegroundColor White
}
else {
    Write-Host "`n⚠️  Some shaders have errors!" -ForegroundColor Red
}