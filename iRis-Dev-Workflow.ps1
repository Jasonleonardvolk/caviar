# D:\Dev\kha\iRis-Dev-Workflow.ps1
# Quick commands for iRis development workflow

Write-Host @"
==================================================
         iRis DEVELOPMENT WORKFLOW
==================================================
"@ -ForegroundColor Cyan

Write-Host "`n🚀 SERVICES RUNNING:" -ForegroundColor Green
Write-Host "  • iRis Dev:  http://localhost:5173" -ForegroundColor White
Write-Host "  • Penrose:   http://127.0.0.1:7401" -ForegroundColor White

Write-Host "`n📁 KEY DIRECTORIES:" -ForegroundColor Yellow
Write-Host "  • UI Components:  D:\Dev\kha\tori_ui_svelte\src\lib\components" -ForegroundColor Gray
Write-Host "  • Routes/Pages:   D:\Dev\kha\tori_ui_svelte\src\routes" -ForegroundColor Gray
Write-Host "  • API Routes:     D:\Dev\kha\tori_ui_svelte\src\routes\api" -ForegroundColor Gray
Write-Host "  • Penrose API:    D:\Dev\kha\services\penrose\main.py" -ForegroundColor Gray

Write-Host "`n🔧 HOT RELOAD:" -ForegroundColor Cyan
Write-Host "  • Vite auto-reloads on file changes" -ForegroundColor White
Write-Host "  • Penrose requires restart for changes" -ForegroundColor White

Write-Host "`n📝 COMMON TASKS:" -ForegroundColor Magenta

Write-Host "`n  1. Create new page:" -ForegroundColor White
Write-Host "     Create: src\routes\mypage\+page.svelte" -ForegroundColor Gray
Write-Host "     Access: http://localhost:5173/mypage" -ForegroundColor Gray

Write-Host "`n  2. Add API endpoint:" -ForegroundColor White
Write-Host "     Create: src\routes\api\myendpoint\+server.ts" -ForegroundColor Gray
Write-Host "     Access: http://localhost:5173/api/myendpoint" -ForegroundColor Gray

Write-Host "`n  3. Test device tiers:" -ForegroundColor White
Write-Host "     Open DevTools (F12)" -ForegroundColor Gray
Write-Host "     Toggle device mode (Ctrl+Shift+M)" -ForegroundColor Gray
Write-Host "     Select iPhone/Android device" -ForegroundColor Gray
Write-Host "     Visit: http://localhost:5173/device/matrix" -ForegroundColor Gray

Write-Host "`n  4. Test Penrose integration:" -ForegroundColor White
Write-Host "     Direct: http://127.0.0.1:7401/docs" -ForegroundColor Gray
Write-Host "     Proxy:  http://localhost:5173/api/penrose/docs" -ForegroundColor Gray

Write-Host "`n⚡ QUICK ACTIONS:" -ForegroundColor Yellow
$action = Read-Host "`nSelect action: [T]est all, [O]pen dashboard, [C]heck status, [S]kip"

switch ($action.ToUpper()) {
    "T" {
        Write-Host "`nRunning tests..." -ForegroundColor Green
        & "D:\Dev\kha\Test-Everything-Now.ps1"
    }
    "O" {
        Write-Host "`nOpening dashboard..." -ForegroundColor Green
        & "D:\Dev\kha\Open-iRis-Dashboard.ps1"
    }
    "C" {
        Write-Host "`nChecking status..." -ForegroundColor Green
        & "D:\Dev\kha\Check-Port-Processes.ps1"
    }
    default {
        Write-Host "`nHappy coding! 🚀" -ForegroundColor Cyan
    }
}