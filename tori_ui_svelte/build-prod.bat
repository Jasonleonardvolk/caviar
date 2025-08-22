@echo off
echo 🏗️ Building TORI for Production...
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo 📦 Installing dependencies...
    call npm install
    echo.
)

echo 🔨 Building optimized bundle...
call npm run build

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ TORI Build Complete!
    echo 📁 Output: ./build/
    echo 🚀 Ready for deployment
    echo.
    
    REM Optionally start preview server
    set /p startPreview="Start preview server? (y/n): "
    if /i "%startPreview%"=="y" (
        echo.
        echo 🌐 Starting preview server...
        call npm run preview
    )
) else (
    echo.
    echo ❌ Build failed! Check the output above for errors.
    pause
)