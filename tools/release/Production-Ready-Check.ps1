param([string]$ProjectRoot = "D:\Dev\kha")

Write-Host "`n🚀 PRODUCTION READINESS CHECK" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$checks = @{
    "Stripe Key" = Test-Path "D:\Dev\kha\frontend\.env"
    "Snap Guides" = Test-Path "D:\Dev\kha\integrations\snap\guides\*.md"
    "TikTok Guides" = Test-Path "D:\Dev\kha\integrations\tiktok\guides\*.md"
    "GLB Files" = Test-Path "D:\Dev\kha\exports\templates\*.glb"
    "KTX2 Files" = Test-Path "D:\Dev\kha\exports\textures_ktx2\*.ktx2"
    "Export Directory" = Test-Path "D:\Dev\kha\exports\bundles"
}

$allGood = $true
foreach($item in $checks.GetEnumerator()) {
    if ($item.Value) {
        Write-Host "✅ $($item.Key)" -ForegroundColor Green
    } else {
        Write-Host "❌ $($item.Key)" -ForegroundColor Red
        $allGood = $false
    }
}

Write-Host "`n================================" -ForegroundColor Cyan

if ($allGood) {
    Write-Host "✅ READY FOR PRODUCTION!" -ForegroundColor Green
    Write-Host "`n⚠️  IMPORTANT: Restart the dev server to load the new .env file:" -ForegroundColor Yellow
    Write-Host "   1. Press Ctrl+C in the dev server terminal" -ForegroundColor White
    Write-Host "   2. Run: cd frontend && pnpm.cmd dev" -ForegroundColor White
    Write-Host "`nThen run the health check again:" -ForegroundColor Cyan
    Write-Host "   powershell -ExecutionPolicy Bypass -File .\tools\release\Verify-Health.ps1" -ForegroundColor White
} else {
    Write-Host "❌ NOT READY - Fix the issues above" -ForegroundColor Red
}