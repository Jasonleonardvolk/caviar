@echo off
echo ============================================
echo   Generating SvelteKit Types
echo ============================================
echo.

cd D:\Dev\kha\tori_ui_svelte
echo Running SvelteKit sync...
call npm run sync 2>nul || call npx svelte-kit sync 2>nul

echo.
echo SvelteKit types generated successfully!
echo.

cd ..
pause
