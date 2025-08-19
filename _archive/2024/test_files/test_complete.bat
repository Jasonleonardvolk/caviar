@echo off
echo.
echo ========================================
echo TORI Complete AV Fix Test
echo ========================================
echo.
echo This test includes ALL necessary av submodules
echo including av.video.frame.VideoFrame.pict_type
echo.

python test_entropy_complete.py

echo.
echo ========================================
echo If successful, run: python tori_launcher_fixed.py
echo ========================================
echo.
pause
