@echo off
cls
echo ==================================================
echo       iRis LAUNCH DAY - QUICK START
echo ==================================================
echo.

echo [1] Checking readiness...
echo.

if exist "site\showcase\A_shock_proof.mp4" (
    echo   [OK] Video A ready
) else (
    echo   [!!] Video A missing
)

if exist "site\showcase\B_how_to_60s.mp4" (
    echo   [OK] Video B ready
) else (
    echo   [!!] Video B missing
)

if exist "site\showcase\C_buyers_clip.mp4" (
    echo   [OK] Video C ready
) else (
    echo   [!!] Video C missing
)

echo.
echo [2] Opening launch pages...
start http://localhost:5173
start http://localhost:5173/hologram-studio
start http://localhost:5173/pricing

echo.
echo ==================================================
echo           LAUNCH CHECKLIST
echo ==================================================
echo.
echo [ ] Update landing page with new copy
echo [ ] Test Free recording (10s + watermark)
echo [ ] Test Plus recording (60s, no watermark)
echo [ ] Post to X/Twitter with Video A
echo [ ] Post to Instagram with Video B
echo [ ] Post to TikTok with Video C
echo [ ] Share #HologramDrop challenge
echo [ ] Monitor Stripe dashboard
echo [ ] Engage with comments
echo [ ] Thank early adopters
echo.
echo ==================================================
echo            QUICK COMMANDS
echo ==================================================
echo.
echo Update landing:
echo   copy /y tori_ui_svelte\src\routes\+page.svelte.new tori_ui_svelte\src\routes\+page.svelte
echo.
echo Commit changes:
echo   git add . ^&^& git commit -m "Launch day" ^&^& git push
echo.
echo ==================================================
echo.
echo Ready to launch? GO GET 'EM! 
echo.
pause