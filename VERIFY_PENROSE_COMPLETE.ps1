# Penrose Integration - Complete Verification
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Penrose Integration - Final Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Always start from project root
Set-Location C:\Users\jason\Desktop\tori\kha

# Activate venv
& .\.venv\Scripts\Activate.ps1

$allPassed = $true

# Test 1: Wheel Import
Write-Host "[1/3] Testing Wheel Import in venv..." -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Yellow
try {
    $result = & .\.venv\Scripts\python -c "import penrose_engine_rs, sys; print(f'✅ Rust backend from {sys.executable}')" 2>&1
    Write-Host $result -ForegroundColor Green
} catch {
    Write-Host "❌ FAILED: Could not import penrose_engine_rs" -ForegroundColor Red
    $allPassed = $false
}
Write-Host ""

# Test 2: Performance Test
Write-Host "[2/3] Running Penrose Performance Test..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

# Create test if it doesn't exist
if (-not (Test-Path "tests\test_penrose.py")) {
    New-Item -ItemType Directory -Force -Path tests | Out-Null
    @'
import penrose_engine_rs
import time
import numpy as np

def test_penrose_performance():
    # Test single similarity
    v1 = [1.0] * 512
    v2 = [1.0] * 512
    
    start = time.perf_counter()
    result = penrose_engine_rs.compute_similarity(v1, v2)
    elapsed = time.perf_counter() - start
    
    print(f'\nSingle similarity: {elapsed*1000:.3f}ms')
    assert elapsed < 0.003  # Should be under 3ms
    assert abs(result - 1.0) < 0.001  # Should be ~1.0
    
    # Test batch similarity
    corpus = [list(np.random.rand(512)) for _ in range(1000)]
    query = list(np.random.rand(512))
    
    start = time.perf_counter()
    results = penrose_engine_rs.batch_similarity(query, corpus)
    elapsed = time.perf_counter() - start
    
    print(f'Batch similarity (1000): {elapsed*1000:.3f}ms')
    assert elapsed < 0.050  # Should be under 50ms for 1000
    print(f'✅ Performance tests passed!')
'@ | Out-File -FilePath tests\test_penrose.py -Encoding UTF8
}

# Install pytest if needed
& .\.venv\Scripts\python -m pip install pytest numpy --quiet

# Run test
$testResult = & .\.venv\Scripts\python -m pytest tests\test_penrose.py -v 2>&1 | Out-String
if ($testResult -match "passed") {
    Write-Host "✅ Performance test passed!" -ForegroundColor Green
    Write-Host $testResult
} else {
    Write-Host "❌ Performance test failed!" -ForegroundColor Red
    $allPassed = $false
}
Write-Host ""

# Test 3: Full App
Write-Host "[3/3] Testing Full App with Rust Requirement..." -ForegroundColor Yellow
Write-Host "==============================================" -ForegroundColor Yellow

# Start launcher in background
$launcherLog = "launcher_test.log"
$process = Start-Process -FilePath .\.venv\Scripts\python -ArgumentList "enhanced_launcher.py", "--require-penrose", "--no-browser" -RedirectStandardOutput $launcherLog -RedirectStandardError launcher_error.log -PassThru -WindowStyle Hidden

# Wait for initialization
Start-Sleep -Seconds 5

# Check if process started successfully
if (-not $process.HasExited) {
    Stop-Process -Id $process.Id -Force
    
    # Check log for Rust initialization
    $logContent = Get-Content $launcherLog -ErrorAction SilentlyContinue
    if ($logContent -match "Penrose engine initialized \(rust\)") {
        Write-Host "✅ App boots with Rust backend!" -ForegroundColor Green
        $logContent | Select-String "Penrose" | ForEach-Object { Write-Host $_ -ForegroundColor Gray }
    } else {
        Write-Host "❌ FAILED: Launcher did not initialize with Rust backend" -ForegroundColor Red
        Write-Host "Log content:" -ForegroundColor Yellow
        $logContent | ForEach-Object { Write-Host $_ }
        $allPassed = $false
    }
} else {
    Write-Host "❌ FAILED: Launcher exited immediately" -ForegroundColor Red
    if (Test-Path launcher_error.log) {
        Write-Host "Error log:" -ForegroundColor Yellow
        Get-Content launcher_error.log
    }
    $allPassed = $false
}

# Cleanup
Remove-Item $launcherLog -ErrorAction SilentlyContinue
Remove-Item launcher_error.log -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($allPassed) {
    Write-Host "✅ ALL LOCAL TESTS PASSED!" -ForegroundColor Green
} else {
    Write-Host "❌ SOME TESTS FAILED!" -ForegroundColor Red
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Push commits if not already done:" -ForegroundColor White
Write-Host "   git push" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Check GitHub Actions:" -ForegroundColor White
Write-Host "   https://github.com/Jasonleonardvolk/Tori/actions" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Look for green 'Build Penrose Wheels' workflow" -ForegroundColor White
Write-Host ""
Write-Host "4. For fresh clone test, run in a temp folder:" -ForegroundColor White
Write-Host '   git clone --depth 1 https://github.com/Jasonleonardvolk/Tori tori-clean' -ForegroundColor Gray
Write-Host '   cd tori-clean' -ForegroundColor Gray
Write-Host '   python -m venv .venv && .\.venv\Scripts\activate' -ForegroundColor Gray
Write-Host '   pip install -r requirements-dev.txt' -ForegroundColor Gray
Write-Host '   cd concept_mesh\penrose_rs && maturin develop --release' -ForegroundColor Gray
Write-Host '   cd ..\.. && python enhanced_launcher.py --require-penrose --no-browser' -ForegroundColor Gray
Write-Host ""

if ($allPassed) {
    Write-Host "🎉 If CI is also green, Phase 1 is COMPLETE! 🎉" -ForegroundColor Green
}

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
