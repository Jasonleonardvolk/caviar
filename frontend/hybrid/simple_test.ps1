# simple_test.ps1
# Simple test of fixed WGSL shaders

Write-Host ""
Write-Host "==== SIMPLE WGSL TEST ====" -ForegroundColor Cyan
Write-Host ""

$original = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\lightFieldComposer.wgsl"
$enhanced = "C:\Users\jason\Desktop\tori\kha\frontend\hybrid\wgsl\lightFieldComposerEnhanced.wgsl"

Write-Host "Testing original shader..." -ForegroundColor Yellow
naga $original
if ($LASTEXITCODE -eq 0) {
    Write-Host "PASS: Original shader is valid" -ForegroundColor Green
} else {
    Write-Host "FAIL: Original shader has errors" -ForegroundColor Red
}

Write-Host ""
Write-Host "Testing enhanced shader..." -ForegroundColor Yellow
naga $enhanced
if ($LASTEXITCODE -eq 0) {
    Write-Host "PASS: Enhanced shader is valid" -ForegroundColor Green
} else {
    Write-Host "FAIL: Enhanced shader has errors" -ForegroundColor Red
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan
