Write-Host "`n🔧 Fixing Entropy Pruning Import Error" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Navigate to the correct directory
Set-Location -Path "C:\Users\jason\Desktop\tori\kha"

Write-Host "`nCurrent directory: $(Get-Location)" -ForegroundColor Yellow

# Check if entropy_prune.py exists
if (Test-Path "ingest_pdf\entropy_prune.py") {
    Write-Host "✅ entropy_prune.py exists" -ForegroundColor Green
} else {
    Write-Host "❌ entropy_prune.py NOT FOUND!" -ForegroundColor Red
    Write-Host "Please restore or create this file first." -ForegroundColor Yellow
    exit 1
}

# Run the fix
Write-Host "`nRunning import fix..." -ForegroundColor Yellow
python fix_entropy_import.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Import fix completed successfully" -ForegroundColor Green
    
    # Run the tests
    Write-Host "`nRunning tests..." -ForegroundColor Yellow
    python test_entropy_pruning.py
} else {
    Write-Host "`n❌ Import fix failed" -ForegroundColor Red
}

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "Done!" -ForegroundColor Green
