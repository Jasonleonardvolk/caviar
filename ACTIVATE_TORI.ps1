# TORI Development Environment Setup
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TORI Development Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Always start from project root
Set-Location C:\Users\jason\Desktop\tori\kha

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Verify activation
Write-Host "Environment activated!" -ForegroundColor Green
Write-Host "Python location: $(Get-Command python).Path" -ForegroundColor Yellow
Write-Host ""

# Quick Penrose test
Write-Host "Testing Penrose import..." -ForegroundColor Yellow
python -c "import penrose_engine_rs, sys; print('✅ Penrose Rust backend ready from', sys.executable)"

Write-Host ""
Write-Host "Ready for development!" -ForegroundColor Green
Write-Host "Common commands:" -ForegroundColor Cyan
Write-Host "  python enhanced_launcher.py              # Run TORI" -ForegroundColor White
Write-Host "  python enhanced_launcher.py --no-browser # Run without browser" -ForegroundColor White
Write-Host "  cd concept_mesh\penrose_rs && maturin develop --release  # Rebuild Rust" -ForegroundColor White
