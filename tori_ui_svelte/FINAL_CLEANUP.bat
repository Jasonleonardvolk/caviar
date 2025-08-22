@echo off
echo.
echo ========================================
echo   FINAL CLEANUP - TORI PostCSS Fix
echo ========================================
echo.

echo 🧹 Step 1: Deleting Vite preprocess temp files...
for /r . %%f in (*.vite-preprocess.css) do (
    if exist "%%f" (
        del "%%f"
        echo   🗑️ Deleted: %%~nxf
    )
)

echo.
echo 🧹 Step 2: Clearing Vite cache...
if exist "node_modules\.vite" (
    rmdir /s /q "node_modules\.vite"
    echo   ✅ Vite cache cleared
) else (
    echo   ℹ️ No Vite cache to clear
)

echo.
echo 🧹 Step 3: Clearing SvelteKit cache...
if exist ".svelte-kit" (
    rmdir /s /q ".svelte-kit"
    echo   ✅ SvelteKit cache cleared
) else (
    echo   ℹ️ No SvelteKit cache to clear
)

echo.
echo 🔍 Step 4: Verifying PostCSS config...
if exist "postcss.config.js" (
    echo   ✅ PostCSS config exists
) else (
    echo   ❌ PostCSS config missing - creating simple one...
    echo export default { plugins: { tailwindcss: {}, autoprefixer: {} } }; > postcss.config.js
    echo   ✅ PostCSS config created
)

echo.
echo 🧹 Step 5: Running comprehensive cleanup...
node final-cleanup.js

echo.
echo ========================================
echo 🎉 FINAL CLEANUP COMPLETED!
echo ========================================
echo.
echo ✅ All temp files deleted
echo ✅ All caches cleared  
echo ✅ Files cleaned and verified
echo.
echo 🚀 Your PostCSS error should now be completely resolved!
echo.
echo Next: Run 'npm run dev' to test!
echo.
pause
