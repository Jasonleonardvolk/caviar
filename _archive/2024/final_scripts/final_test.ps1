Write-Host ""
Write-Host "TORI FINAL TEST - Complete Fix" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green
Write-Host ""
Write-Host "Great news! Entropy pruning is ALREADY WORKING!" -ForegroundColor Yellow
Write-Host "This test adds AVError for full compatibility." -ForegroundColor Yellow
Write-Host ""

# Run the test
python test_entropy_with_averror.py

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  - If you see 'COMPLETE SUCCESS', run: python tori_launcher_with_averror.py" -ForegroundColor White
Write-Host "  - Or just use: python enhanced_launcher.py (entropy pruning works!)" -ForegroundColor White
Write-Host ""
