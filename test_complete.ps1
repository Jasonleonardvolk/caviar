Write-Host ""
Write-Host "TORI Complete AV Fix Test" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This creates a COMPLETE av mock with all submodules" -ForegroundColor Yellow
Write-Host "including the critical av.video.frame.VideoFrame.pict_type" -ForegroundColor Yellow
Write-Host ""

# Run the complete test
python test_entropy_complete.py

Write-Host ""
Write-Host "If you see 'ENTROPY PRUNING IS FULLY FUNCTIONAL!' above," -ForegroundColor Green
Write-Host "then run: python tori_launcher_fixed.py" -ForegroundColor Green
Write-Host ""
