# Quick Fix: Ensure Python Fallback Works
Write-Host "Quick Fix: Configuring Penrose Python Fallback" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# This ensures TORI runs with the Python fallback while we fix the Rust build

$projectRoot = $PSScriptRoot

Write-Host "`n[1] Checking current Penrose setup..." -ForegroundColor Yellow

# Check if similarity module exists
$similarityPath = Join-Path $projectRoot "concept_mesh\similarity\__init__.py"
if (Test-Path $similarityPath) {
    Write-Host "[OK] Similarity module exists" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Similarity module missing - TORI may have issues" -ForegroundColor Yellow
}

# Ensure Python packages are installed
Write-Host "`n[2] Installing required Python packages..." -ForegroundColor Yellow
pip install numpy scipy scikit-learn numba --upgrade

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Python packages installed" -ForegroundColor Green
    
    # Test the fallback
    Write-Host "`n[3] Testing Python fallback..." -ForegroundColor Yellow
    
    $testScript = @'
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from concept_mesh.similarity import penrose_available, compute_similarity
    import numpy as np
    
    print(f"[INFO] Penrose available: {penrose_available}")
    
    # Test similarity computation
    embeddings = np.random.rand(100, 64)
    queries = np.random.rand(5, 64)
    
    import time
    start = time.perf_counter()
    similarities = compute_similarity(embeddings, queries, threshold=0.7)
    elapsed = time.perf_counter() - start
    
    print(f"[OK] Similarity computation works!")
    print(f"[INFO] Computed {similarities.shape} similarities in {elapsed*1000:.1f}ms")
    print(f"[INFO] Using: {'Rust engine' if penrose_available else 'Python fallback'}")
    
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'@
    
    $testScript | python
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[SUCCESS] Python fallback is working!" -ForegroundColor Green
        Write-Host "`nYou can now run TORI with:" -ForegroundColor Cyan
        Write-Host "  python enhanced_launcher.py" -ForegroundColor White
        Write-Host "`nThe system will use the Python implementation until Rust build is fixed." -ForegroundColor Gray
    }
} else {
    Write-Host "[ERROR] Failed to install Python packages" -ForegroundColor Red
}

Write-Host "`n[4] Build Options:" -ForegroundColor Yellow
Write-Host "`nOption A: Fix Rust build (fastest)" -ForegroundColor White
Write-Host "  1. Run DIAGNOSE_PENROSE_BUILD.bat to see what's missing" -ForegroundColor Gray
Write-Host "  2. Install Visual Studio Build Tools if needed" -ForegroundColor Gray
Write-Host "  3. Run BUILD_PENROSE.bat again" -ForegroundColor Gray

Write-Host "`nOption B: Use Numba acceleration (good performance, no C++)" -ForegroundColor White
Write-Host "  1. Run INSTALL_PENROSE_NUMBA.bat" -ForegroundColor Gray
Write-Host "  2. Get 10-20x speedup without Rust" -ForegroundColor Gray

Write-Host "`nOption C: Continue with Python fallback (works now)" -ForegroundColor White
Write-Host "  1. Just run: python enhanced_launcher.py" -ForegroundColor Gray
Write-Host "  2. Slower but fully functional" -ForegroundColor Gray

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")