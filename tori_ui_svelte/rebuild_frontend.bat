@echo off
echo ========================================
echo Rebuilding TORI Frontend
echo ========================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tori_ui_svelte

echo [1/3] Cleaning cache...
if exist .svelte-kit (
    rmdir /s /q .svelte-kit
    echo Cache cleared
)

echo.
echo [2/3] Installing dependencies...
call npm install

echo.
echo [3/3] Building frontend...
call npm run build

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo To start development server: npm run dev
echo.
pause
