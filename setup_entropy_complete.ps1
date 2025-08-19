Write-Host "`n🚀 Complete Entropy Pruning Setup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Navigate to the correct directory
Set-Location -Path "C:\Users\jason\Desktop\tori\kha"

Write-Host "`n📍 Current directory: $(Get-Location)" -ForegroundColor Yellow

# Step 1: Install dependencies
Write-Host "`n1️⃣ Installing required dependencies..." -ForegroundColor Green
python install_entropy_dependencies.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Dependency installation failed!" -ForegroundColor Red
    Write-Host "Try running in your virtual environment:" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\activate" -ForegroundColor Yellow
    Write-Host "  pip install sentence-transformers scikit-learn numpy" -ForegroundColor Yellow
    exit 1
}

# Step 2: Verify installation
Write-Host "`n2️⃣ Verifying entropy pruning setup..." -ForegroundColor Green
python verify_entropy.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Entropy pruning is fully operational!" -ForegroundColor Green
    
    # Step 3: Run full test suite
    Write-Host "`n3️⃣ Running comprehensive tests..." -ForegroundColor Green
    python test_entropy_pruning.py
} else {
    Write-Host "`n⚠️  Some issues remain. Check the error messages above." -ForegroundColor Yellow
}

Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green

# Quick test command
Write-Host "`n📋 Quick test command:" -ForegroundColor Yellow
Write-Host '  python -c "from ingest_pdf.pipeline.pruning import apply_entropy_pruning; print(''✅ Import successful!'')"' -ForegroundColor White
