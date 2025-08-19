Write-Host "`n🚀 Quick Workaround for Entropy Pruning" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

Write-Host "`nThis will install sentence-transformers without touching scipy" -ForegroundColor Yellow

# Just install sentence-transformers with pip
Write-Host "`nInstalling sentence-transformers..." -ForegroundColor Green
pip install sentence-transformers --no-deps

Write-Host "`nInstalling missing dependencies..." -ForegroundColor Green
pip install transformers
pip install huggingface-hub

# Test the import
Write-Host "`n🧪 Testing import..." -ForegroundColor Yellow
python -c "from sentence_transformers import SentenceTransformer; print('✅ Import successful!')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Success! Now test entropy pruning:" -ForegroundColor Green
    Write-Host "  python verify_entropy.py" -ForegroundColor White
} else {
    Write-Host "`n❌ Import test failed" -ForegroundColor Red
}
