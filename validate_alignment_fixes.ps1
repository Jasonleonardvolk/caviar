Write-Host "`n🎯 FINAL WGSL ALIGNMENT FIX VALIDATION" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Validating uniform buffer alignment fixes...`n" -ForegroundColor Yellow

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$allShaders = Get-ChildItem -Path $shaderDir -Filter "*.wgsl" | Sort-Object Name

$results = @{
    Valid = @()
    Invalid = @()
}

foreach ($shader in $allShaders) {
    Write-Host "▶ Checking $($shader.Name) ..." -NoNewline
    
    $result = & naga $shader.FullName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅" -ForegroundColor Green
        $results.Valid += $shader.Name
    }
    else {
        Write-Host " ❌" -ForegroundColor Red
        $results.Invalid += @{
            Name = $shader.Name
            Error = $result | Out-String
        }
    }
}

Write-Host "`n📊 FINAL RESULTS" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host "✅ Valid shaders: $($results.Valid.Count) / $($allShaders.Count)" -ForegroundColor Green

if ($results.Invalid.Count -eq 0) {
    Write-Host "`n🎉 ALL SHADERS ARE NOW VALID! 🎉" -ForegroundColor Green
    Write-Host "`n✅ All alignment issues have been fixed!" -ForegroundColor Green
    Write-Host "`n📋 Fixed issues:" -ForegroundColor Yellow
    Write-Host "  • lenticularInterlace.wgsl: Changed array<f32, 8> to array<vec4<f32>, 2>" -ForegroundColor White
    Write-Host "  • velocityField.wgsl: Changed array<f32, 16> to array<vec4<f32>, 4>" -ForegroundColor White
    Write-Host "`n🚀 Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Bundle shaders: npx tsx scripts/bundleShaders.ts" -ForegroundColor White
    Write-Host "  2. Launch app: .\START_TORI_HARDENED.bat -Force" -ForegroundColor White
}
else {
    Write-Host "`n❌ Still have invalid shaders:" -ForegroundColor Red
    foreach ($invalid in $results.Invalid) {
        Write-Host "`n$($invalid.Name):" -ForegroundColor Yellow
        Write-Host $invalid.Error -ForegroundColor Red
    }
}

# Save detailed log
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logPath = "C:\Users\jason\Desktop\tori\kha\alignment_fix_validation_$timestamp.log"
@{
    Timestamp = $timestamp
    TotalShaders = $allShaders.Count
    ValidShaders = $results.Valid
    InvalidShaders = $results.Invalid
} | ConvertTo-Json -Depth 10 | Out-File -FilePath $logPath -Encoding UTF8

Write-Host "`nDetailed log saved to: $logPath" -ForegroundColor Cyan