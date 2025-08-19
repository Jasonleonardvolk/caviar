# Test if 't' function is available
Write-Host "🔍 Testing TORI 't' shortcut..." -ForegroundColor Cyan
Write-Host ""

# Check if function exists
if (Get-Command t -ErrorAction SilentlyContinue) {
    Write-Host "✅ 't' function is available!" -ForegroundColor Green
    Write-Host ""
    
    # Show what it will do
    Write-Host "📝 When you type 't', it will:" -ForegroundColor Cyan
    Write-Host "   1. Navigate to: C:\Users\jason\Desktop\tori\kha" -ForegroundColor White
    Write-Host "   2. Activate: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "   3. Show ready message" -ForegroundColor White
    Write-Host ""
    
    # Check if alias exists
    if (Get-Alias tori -ErrorAction SilentlyContinue) {
        Write-Host "✅ 'tori' alias also available!" -ForegroundColor Green
    }
    
    Write-Host "🚀 You're all set! Type 't' to jump into TORI development!" -ForegroundColor Yellow
} else {
    Write-Host "❌ 't' function not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "📝 To set it up:" -ForegroundColor Yellow
    Write-Host "   1. Run: .\SETUP_T_SHORTCUT.bat" -ForegroundColor White
    Write-Host "   2. Open a NEW PowerShell window" -ForegroundColor White
    Write-Host "   3. Try again!" -ForegroundColor White
}

Write-Host ""
Write-Host "Profile location: $($PROFILE)" -ForegroundColor Gray
