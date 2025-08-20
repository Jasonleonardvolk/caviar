@echo off
cls
echo ==================================================
echo     iRis PHASE 4 - QUICK VIDEO RECORDER
echo ==================================================
echo.

echo VIDEO A: SHOCK PROOF (10s Free with watermark)
echo ------------------------------------------------
echo 1. Open hologram studio
echo 2. Use FREE plan (watermark visible)
echo 3. Record 10 seconds
echo 4. Save as: A_shock_proof.mp4
echo.
pause

start http://localhost:5173/hologram-studio

echo.
echo VIDEO B: HOW-TO (20-30s Plus tutorial)
echo ------------------------------------------------
echo 1. Go to /pricing
echo 2. Upgrade to Plus
echo 3. Record 20-30s tutorial
echo 4. Save as: B_how_to_60s.mp4
echo.
pause

start http://localhost:5173/pricing

echo.
echo VIDEO C: BUYERS CLIP (15-20s deliverables)
echo ------------------------------------------------
echo 1. Open File Explorer
echo 2. Show D:\Dev\kha\exports\video\
echo 3. Play one video file
echo 4. Save as: C_buyers_clip.mp4
echo.
pause

explorer D:\Dev\kha\exports\video

echo.
echo ==================================================
echo            FINAL LOCATIONS
echo ==================================================
echo.
echo Copy your videos to:
echo   D:\Dev\kha\site\showcase\A_shock_proof.mp4
echo   D:\Dev\kha\site\showcase\B_how_to_60s.mp4
echo   D:\Dev\kha\site\showcase\C_buyers_clip.mp4
echo.
echo Also upload to:
echo   Google Drive\iRis\Showcase\
echo.
pause