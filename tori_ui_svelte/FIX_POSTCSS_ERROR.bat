@echo off
echo.
echo ========================================
echo   TORI PostCSS Error Fix
echo ========================================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte"

echo 🔧 Running PostCSS error fix...
node fix-postcss-error.js

echo.
echo ✅ Fix completed! 
echo.
echo 🚀 Now run: npm run dev
echo.
pause
