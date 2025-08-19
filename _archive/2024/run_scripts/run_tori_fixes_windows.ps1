# Windows PowerShell script to run TORI fixes and verification

Write-Host "🔧 TORI Fix and Verification Script for Windows" -ForegroundColor Cyan
Write-Host ("=" * 50)

# Step 1: Apply fixes
Write-Host "`n📝 Applying fixes..." -ForegroundColor Yellow
python fix_tori_automatic_v3.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Fix script failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Run unit tests (optional)
if (Test-Path "tests\test_launch_order.py") {
    Write-Host "`n🧪 Running unit tests..." -ForegroundColor Yellow
    pytest tests\test_launch_order.py
}

# Step 3: Start TORI in background
Write-Host "`n🚀 Starting TORI in background..." -ForegroundColor Yellow
$toriProcess = Start-Process -FilePath "poetry" -ArgumentList "run", "python", "enhanced_launcher.py" -WindowStyle Hidden -PassThru

# Step 4: Wait for startup
Write-Host "⏳ Waiting 15 seconds for TORI to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Step 5: Verify
Write-Host "`n🔍 Running verification..." -ForegroundColor Yellow
python verify_tori_fixes_v3.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ TORI is running successfully!" -ForegroundColor Green
    Write-Host "🌐 Open your browser to: http://localhost:5173" -ForegroundColor Cyan
    Write-Host "📚 API docs at: http://localhost:8002/docs" -ForegroundColor Cyan
    
    # Keep script running
    Write-Host "`nPress Ctrl+C to stop TORI..." -ForegroundColor Yellow
    try {
        while ($true) {
            Start-Sleep -Seconds 1
        }
    }
    catch {
        Write-Host "`n👋 Shutting down TORI..." -ForegroundColor Yellow
        if ($toriProcess -and !$toriProcess.HasExited) {
            Stop-Process -Id $toriProcess.Id -Force
        }
    }
}
else {
    Write-Host "`n❌ Verification failed!" -ForegroundColor Red
    Write-Host "Check the logs above for details." -ForegroundColor Red
    if ($toriProcess -and !$toriProcess.HasExited) {
        Stop-Process -Id $toriProcess.Id -Force
    }
}
