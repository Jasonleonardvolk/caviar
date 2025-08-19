@echo off
REM Quick Start Script for Enhanced TORI Holographic System (Windows)

echo ========================================
echo  TORI Holographic Enhancement Quick Start
echo ========================================
echo.

REM Check Node.js
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo [OK] Node.js found
echo.

REM Run integration
echo Starting integration...
node integrate_enhancements.js

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Integration successful!
    echo.
    
    REM Offer to start dev server
    set /p response="Would you like to start the development server? (y/n): "
    
    if /i "%response%"=="y" (
        cd ..\tori_ui_svelte
        echo Starting development server...
        npm run dev
    ) else (
        echo.
        echo To start manually:
        echo   cd ..\tori_ui_svelte
        echo   npm run dev
    )
) else (
    echo.
    echo [ERROR] Integration failed!
    echo Please check the error messages above.
)

pause
