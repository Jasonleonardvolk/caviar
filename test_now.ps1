Write-Host ""
Write-Host "TORI Quick Entropy Test" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host ""

Write-Host "The av.py mock module has been created to fix compatibility issues." -ForegroundColor Green
Write-Host ""

Write-Host "Running entropy pruning test..." -ForegroundColor Yellow
python test_entropy_direct.py

Write-Host ""
Write-Host "Running comprehensive system check..." -ForegroundColor Yellow
python final_system_check.py

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
