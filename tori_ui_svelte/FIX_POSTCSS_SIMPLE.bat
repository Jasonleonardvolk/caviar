@echo off
echo.
echo ========================================
echo   TORI PostCSS Error Fix
echo ========================================
echo.

echo 🔧 Step 1: Installing missing dependencies...
npm install -D tailwindcss@latest autoprefixer@latest postcss@latest

echo.
echo 🧹 Step 2: Clearing Vite cache...
if exist "node_modules\.vite" (
    rmdir /s /q "node_modules\.vite"
    echo ✅ Vite cache cleared
) else (
    echo ℹ️ No Vite cache to clear
)

echo.
echo 🛠️ Step 3: Creating simple PostCSS config...
echo /** Simple PostCSS Config */> postcss.config.simple.js
echo export default { plugins: { tailwindcss: {}, autoprefixer: {} } };>> postcss.config.simple.js

echo.
echo 📋 Creating backup of current config...
if exist "postcss.config.js" (
    copy "postcss.config.js" "postcss.config.js.backup"
    echo ✅ Backup created
)

echo.
echo 🔄 Updating PostCSS config...
copy "postcss.config.simple.js" "postcss.config.js"
del "postcss.config.simple.js"

echo.
echo ========================================
echo 🎉 BASIC FIX COMPLETED!
echo ========================================
echo.
echo 🚀 Next steps:
echo   1. Stop your dev server (Ctrl+C)
echo   2. Run: npm run dev
echo   3. Your PostCSS error should be resolved!
echo.
echo If issues persist, run the Node.js fixer:
echo   node fix-postcss-error.js
echo.
pause
