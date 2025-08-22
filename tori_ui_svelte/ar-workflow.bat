@echo off
cls
echo =========================================
echo    AR ASSETS COMPLETE WORKFLOW GUIDE
echo =========================================
echo.
echo Current Status: System 100%% Ready, Models 0%% Downloaded
echo.

:menu
echo What would you like to do?
echo.
echo   1. Check current status
echo   2. Open download pages in browser
echo   3. Verify downloads after downloading
echo   4. Process downloaded models
echo   5. Test complete system
echo   6. Start dev server
echo   7. View full documentation
echo   8. Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto status
if "%choice%"=="2" goto download
if "%choice%"=="3" goto verify
if "%choice%"=="4" goto process
if "%choice%"=="5" goto test
if "%choice%"=="6" goto dev
if "%choice%"=="7" goto docs
if "%choice%"=="8" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:status
echo.
echo Checking AR Assets Status...
echo =====================================
call ar-status-check.bat
goto menu

:download
echo.
echo Opening download pages...
echo =====================================
call open-download-pages.bat
goto menu

:verify
echo.
echo Verifying downloaded files...
echo =====================================
call check-downloads.bat
goto menu

:process
echo.
echo Processing 3D models...
echo =====================================
call process-ar-models.bat
goto menu

:test
echo.
echo Testing AR Assets system...
echo =====================================
call test-ar-assets.bat
goto menu

:dev
echo.
echo Starting development server...
echo =====================================
echo After server starts, visit: http://localhost:3000/assets
echo Press Ctrl+C to stop the server
echo.
pnpm dev
goto menu

:docs
echo.
echo Opening documentation...
start notepad DOWNLOAD_PATHWAYS_CONFIRMED.md
goto menu

:end
echo.
echo AR Assets workflow complete!
echo.
pause