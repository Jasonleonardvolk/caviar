@echo off
echo Navigating to tori_ui_svelte directory...
cd /d "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte"

echo.
echo Checking PostCSS dependencies...
npm list tailwindcss autoprefixer postcss

echo.
echo Installing PostCSS dependencies if needed...
npm install -D tailwindcss@latest autoprefixer@latest postcss@latest

echo.
echo Creating simple PostCSS config...
echo export default { plugins: { tailwindcss: {}, autoprefixer: {} } }; > postcss.config.js

echo.
echo Testing dev server...
echo Starting npm run dev...
echo.
npm run dev
