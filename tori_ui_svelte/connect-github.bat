@echo off
REM connect-github.bat - Quick GitHub connection setup

echo ========================================
echo GitHub Repository Connection
echo ========================================
echo.

REM Default values
set GITHUB_USER=Jasonleonardvolk
set REPO_NAME=iRis

echo GitHub User: %GITHUB_USER%
echo Repository: %REPO_NAME%
echo.

REM Check for existing remote
git remote get-url origin >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARN] Remote origin already exists
    echo Current remote:
    git remote get-url origin
    echo.
    choice /C YN /M "Replace existing remote?"
    if errorlevel 2 goto :push_only
    git remote remove origin
    echo [OK] Removed existing remote
)

REM Add remote
echo [INFO] Adding GitHub remote...
git remote add origin https://github.com/%GITHUB_USER%/%REPO_NAME%.git

if %errorlevel% equ 0 (
    echo [OK] Remote added successfully
) else (
    echo [ERROR] Failed to add remote
    pause
    exit /b 1
)

:push_only
echo.
echo ========================================
echo Ready to push to GitHub
echo ========================================
echo.
echo This will push:
echo   - Branch: main
echo   - Tag: iris-internal-v0.1.0
echo   - All commits
echo.

choice /C YN /M "Push to GitHub now?"
if errorlevel 2 goto :skip_push

echo.
echo [INFO] Pushing to GitHub...
git push -u origin main --follow-tags

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Successfully pushed to GitHub!
    echo ========================================
    echo.
    echo Repository URL:
    echo   https://github.com/%GITHUB_USER%/%REPO_NAME%
    echo.
) else (
    echo.
    echo [ERROR] Push failed!
    echo.
    echo If using HTTPS, you need a Personal Access Token.
    echo Create one at: https://github.com/settings/tokens
    echo.
)
goto :end

:skip_push
echo.
echo [INFO] Push cancelled. To push later, run:
echo   git push -u origin main --follow-tags
echo.

:end
pause
