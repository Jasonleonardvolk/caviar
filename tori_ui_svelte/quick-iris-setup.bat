@echo off
REM Quick iRis GitHub Setup
REM Run this after creating the iRis repository on GitHub

echo ========================================
echo iRis GitHub Quick Setup
echo ========================================
echo.
echo Repository: https://github.com/Jasonleonardvolk/iRis
echo.

REM Check if remote exists
git remote get-url origin >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Removing existing remote...
    git remote remove origin
)

REM Add the iRis repository as origin
echo [INFO] Adding iRis repository as origin...
git remote add origin https://github.com/Jasonleonardvolk/iRis.git

REM Verify
echo.
echo [OK] Remote configured:
git remote -v
echo.

REM Push
echo Ready to push to iRis repository!
echo.
echo This will push:
echo   - Branch: main
echo   - Tag: iris-internal-v0.1.0
echo   - All commits and files
echo.

pause

echo.
echo [INFO] Pushing to GitHub...
git push -u origin main --follow-tags

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo SUCCESS! iRis is now on GitHub!
    echo ========================================
    echo.
    echo View your repository at:
    echo https://github.com/Jasonleonardvolk/iRis
    echo.
) else (
    echo.
    echo [ERROR] Push failed!
    echo.
    echo You may need to:
    echo 1. Create a Personal Access Token at:
    echo    https://github.com/settings/tokens
    echo 2. Use the token as your password
    echo.
)

pause
