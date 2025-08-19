Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Testing TORI with API on port 8002" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Run the test with the correct API URL
python test_concept_mesh_e2e.py --api-url=http://localhost:8002

Write-Host ""
Write-Host "Test completed!" -ForegroundColor Green
