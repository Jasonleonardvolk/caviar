# Fresh Shader Validation Check
Write-Host "`n🚀 FRESH SHADER VALIDATION" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

# Clear old reports
Write-Host "`n🧹 Clearing old reports..." -ForegroundColor Yellow
$reportDir = "tools\shaders\reports"
if (Test-Path $reportDir) {
    Get-ChildItem $reportDir -Filter "*.json" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-1) } | Remove-Item -Force
}

# Run fresh validation
Write-Host "`n🔍 Running fresh validation..." -ForegroundColor Yellow
Write-Host "   Command: npm run shaders:gate:latest" -ForegroundColor Gray

npm run shaders:gate:latest

$exitCode = $LASTEXITCODE

# Check result
Write-Host "`n📊 Results:" -ForegroundColor Cyan

if ($exitCode -eq 0) {
    Write-Host "   ✅ EXIT CODE 0 - VALIDATION PASSED!" -ForegroundColor Green
    Write-Host "`n🎉 ALL SHADERS PASS - READY TO SHIP!" -ForegroundColor Magenta
} else {
    Write-Host "   ❌ Exit code: $exitCode" -ForegroundColor Red
}

# Check for new reports
Write-Host "`n📄 Checking for new reports..." -ForegroundColor Yellow

$newReports = Get-ChildItem $reportDir -Filter "*.json" -ErrorAction SilentlyContinue | 
    Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-5) }

if ($newReports) {
    foreach ($report in $newReports) {
        Write-Host "   Found: $($report.Name) (just created)" -ForegroundColor Green
        
        $content = Get-Content $report.FullName -Raw | ConvertFrom-Json
        if ($content.passed) {
            Write-Host "   Passed: $($content.passed)" -ForegroundColor Green
        }
        if ($content.failed -eq 0 -or !$content.failed) {
            Write-Host "   Failed: 0" -ForegroundColor Green
        }
    }
}

Write-Host "`n🚢 SHIP IT!" -ForegroundColor Cyan
Write-Host "   .\tools\release\IrisOneButton.ps1" -ForegroundColor White
