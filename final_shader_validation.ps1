Write-Host "`n🔥 FINAL SHADER VALIDATION" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan
Write-Host "Validating all fixed shaders with naga...`n" -ForegroundColor Yellow

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$targetShaders = @(
    "lenticularInterlace.wgsl",
    "propagation.wgsl", 
    "velocityField.wgsl"
)

$allValid = $true
$results = @()

foreach ($shader in $targetShaders) {
    $shaderPath = Join-Path $shaderDir $shader
    Write-Host "Validating $shader..." -NoNewline
    
    $result = & naga $shaderPath 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅ VALID!" -ForegroundColor Green
        $results += "✅ $shader - Valid"
    } else {
        Write-Host " ❌ ERROR!" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        $results += "❌ $shader - Error: $result"
        $allValid = $false
    }
}

Write-Host "`n📋 SUMMARY" -ForegroundColor Cyan
Write-Host "==========" -ForegroundColor Cyan

foreach ($result in $results) {
    if ($result -like "*✅*") {
        Write-Host $result -ForegroundColor Green
    } else {
        Write-Host $result -ForegroundColor Red
    }
}

if ($allValid) {
    Write-Host "`n🎉 ALL SHADERS ARE VALID! 🎉" -ForegroundColor Green
    Write-Host "`n✅ Next Steps:" -ForegroundColor Cyan
    Write-Host "1. Bundle shaders: npx tsx scripts/bundleShaders.ts" -ForegroundColor White
    Write-Host "2. Launch app: .\START_TORI_HARDENED.bat -Force" -ForegroundColor White
} else {
    Write-Host "`n⚠️  Some shaders still have errors!" -ForegroundColor Red
    Write-Host "Please check the error messages above." -ForegroundColor Yellow
}

# Save results
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logPath = "C:\Users\jason\Desktop\tori\kha\shader_validation_$timestamp.log"
$results | Out-File -FilePath $logPath -Encoding UTF8
Write-Host "`nValidation log saved to: $logPath" -ForegroundColor Cyan