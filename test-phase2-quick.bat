@echo off
echo === iRis Phase 2 Quick Test ===
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo [1] Installing dependencies...
call pnpm install >nul 2>&1
call npm install stripe >nul 2>&1
echo    Done!

echo.
echo [2] Starting dev server...
start /min cmd /c "pnpm dev"
timeout /t 5 /nobreak >nul

echo.
echo [3] Opening test pages...
start http://localhost:5173/hologram-studio
timeout /t 2 /nobreak >nul
start http://localhost:5173/pricing

echo.
echo =====================================
echo TEST CHECKLIST:
echo =====================================
echo.
echo FREE PLAN:
echo   - Record 10s video
echo   - Check for watermark
echo.
echo UPGRADE:
echo   - Click "Get Plus" 
echo   - Use card: 4242 4242 4242 4242
echo   - Expiry: 12/34, CVC: 123
echo.
echo PLUS PLAN:
echo   - Record 60s video
echo   - Verify NO watermark
echo.
echo =====================================
echo.
echo Test Card: 4242 4242 4242 4242
echo.
pause