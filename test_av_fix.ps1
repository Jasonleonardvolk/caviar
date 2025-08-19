Write-Host ""
Write-Host "AV Module Fix - Final Test" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This test creates a complete mock av module in memory" -ForegroundColor Yellow
Write-Host "before importing anything else, ensuring compatibility." -ForegroundColor Yellow
Write-Host ""

# Run the final test
python test_entropy_final.py

Write-Host ""
Write-Host "If you see 'ENTROPY PRUNING IS FULLY FUNCTIONAL!' above," -ForegroundColor Green
Write-Host "then the fix is working!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Start API: python start_tori_fixed.py" -ForegroundColor White
Write-Host "  2. Or: python enhanced_launcher.py" -ForegroundColor White
Write-Host ""
