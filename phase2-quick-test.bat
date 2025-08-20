@echo off
cls
echo ==================================================
echo     iRis PHASE 2 - QUICK TEST RUNNER
echo ==================================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo [1] Installing dependencies...
call pnpm install >nul 2>&1
echo     Done!

echo.
echo [2] Starting dev server...
start /min cmd /c pnpm dev
echo     Waiting for server...
timeout /t 5 /nobreak >nul
echo     Server ready!

echo.
echo [3] Opening test pages...
start http://localhost:5173/hologram
timeout /t 2 /nobreak >nul

echo.
echo ==================================================
echo TEST 1: FREE PLAN
echo ==================================================
echo.
echo   1. Click "Record 10s"
echo   2. Watch countdown
echo   3. Open downloaded video
echo   4. CHECK: Watermark visible in lower-right
echo.
echo Press any key when complete...
pause >nul

echo.
echo ==================================================
echo TEST 2: STRIPE CHECKOUT
echo ==================================================
echo.
echo Opening pricing page...
start http://localhost:5173/pricing
timeout /t 2 /nobreak >nul

echo.
echo   1. Click "Get Plus"
echo   2. Enter test card: 4242 4242 4242 4242
echo   3. Expiry: 12/34, CVC: 123
echo   4. Complete payment
echo   5. Verify redirect to /thank-you
echo.
echo Press any key when complete...
pause >nul

echo.
echo ==================================================
echo TEST 3: PLUS PLAN (NO WATERMARK)
echo ==================================================
echo.
echo Returning to hologram page...
start http://localhost:5173/hologram

echo.
echo   1. Verify "Plus" shows in recorder
echo   2. Click "Record 60s"
echo   3. Stop recording after few seconds
echo   4. CHECK: NO watermark in video
echo.
echo Press any key when complete...
pause >nul

echo.
echo ==================================================
echo                 TEST COMPLETE
echo ==================================================
echo.
echo Test Results Checklist:
echo   [ ] Free plan has watermark
echo   [ ] Stripe checkout works
echo   [ ] Plus plan has NO watermark
echo.
echo All tests passed? You're ready for Phase 3!
echo.
pause