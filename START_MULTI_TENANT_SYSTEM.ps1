# 🏢 TORI Multi-Tenant System - PowerShell Quick Start
# =================================================

Write-Host "🏢 TORI Multi-Tenant System - Quick Start" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📂 Changing to TORI directory..." -ForegroundColor Yellow
Set-Location "C:\Users\jason\Desktop\tori\kha"

Write-Host "📋 Starting TORI Multi-Tenant API Server..." -ForegroundColor Green
Write-Host ""

# Start the API server in background
Start-Process -FilePath "python" -ArgumentList "start_dynamic_api.py" -WindowStyle Normal

Write-Host "⏳ Waiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

Write-Host ""
Write-Host "🧪 Running Multi-Tenant System Tests..." -ForegroundColor Green
Write-Host ""

# Run the test suite
python test_multi_tenant_system.py

Write-Host ""
Write-Host "🎯 Multi-Tenant System Ready!" -ForegroundColor Green
Write-Host ""
Write-Host "📚 Available endpoints:" -ForegroundColor Cyan
Write-Host "   • Health Check:    http://localhost:8002/health" -ForegroundColor White
Write-Host "   • API Docs:        http://localhost:8002/docs" -ForegroundColor White
Write-Host "   • System Stats:    http://localhost:8002/admin/system/stats" -ForegroundColor White
Write-Host ""
Write-Host "📖 Documentation: MULTI_TENANT_COMPLETE_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 To manually start:" -ForegroundColor Yellow
Write-Host "   cd C:\Users\jason\Desktop\tori\kha" -ForegroundColor White
Write-Host "   python start_dynamic_api.py" -ForegroundColor White
Write-Host ""

# Open browser to API docs
Write-Host "🌐 Opening API documentation in browser..." -ForegroundColor Green
Start-Process "http://localhost:8002/docs"

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
