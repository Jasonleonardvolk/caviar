@echo off
cls
echo ==================================================
echo      iRis PHASE 3 - QUICK DEPLOY TO VERCEL
echo ==================================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo [1] Installing Vercel adapter...
call npm install -D @sveltejs/adapter-vercel >nul 2>&1
echo     Done!

echo.
echo [2] Testing build...
call npm run build >nul 2>&1
if %errorlevel% neq 0 (
    echo     Build failed! Fix errors first.
    pause
    exit /b 1
)
echo     Build successful!

echo.
echo [3] Git status...
git status --short

echo.
echo [4] Ready to commit and push
echo.
echo     Commit message: "iRis launch: recorder + pricing + stripe checkout"
echo.
set /p confirm="Proceed with git commit and push? (y/n): "

if /i "%confirm%"=="y" (
    git add .
    git commit -m "iRis launch: recorder + pricing + stripe checkout"
    git push origin main
    echo.
    echo     Pushed to GitHub!
) else (
    echo     Skipped git operations
)

echo.
echo ==================================================
echo             NEXT STEPS - VERCEL
echo ==================================================
echo.
echo 1. Go to: https://vercel.com
echo 2. Click "Add New Project"
echo 3. Import: Jasonleonardvolk/caviar
echo 4. Root Directory: tori_ui_svelte
echo 5. Add these environment variables:
echo.
echo    STRIPE_SECRET_KEY = sk_test_xxx
echo    STRIPE_PRICE_PLUS = price_xxx
echo    STRIPE_PRICE_PRO = price_xxx
echo    STRIPE_SUCCESS_URL = https://app.vercel.app/thank-you
echo    STRIPE_CANCEL_URL = https://app.vercel.app/pricing?canceled=1
echo.
echo 6. Click "Deploy"
echo.
echo ==================================================
echo.
pause