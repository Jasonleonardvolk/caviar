@echo off
REM Fix-GitHooks.bat - Fixes or removes problematic Git hooks

echo ========================================
echo Fixing Git Hooks
echo ========================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

REM Check if .git\hooks exists
if not exist ".git\hooks" (
    echo [INFO] Creating hooks directory...
    mkdir ".git\hooks"
)

REM Remove the broken pre-push hook
if exist ".git\hooks\pre-push" (
    echo [INFO] Removing broken pre-push hook...
    del ".git\hooks\pre-push"
    echo [OK] Removed old pre-push hook
)

REM Create a working pre-push hook
echo [INFO] Creating new pre-push hook...
(
echo #!/bin/sh
echo # Pre-push hook for iRis project
echo echo "Running pre-push checks..."
echo 
echo # Check if the verification script exists
echo if [ -f "tools/release/Verify-EndToEnd.ps1" ]; then
echo     echo "Running end-to-end tests..."
echo     powershell -NoProfile -File "tools\release\Verify-EndToEnd.ps1"
echo     if [ $? -ne 0 ]; then
echo         echo "End-to-end verification failed. Use --no-verify to bypass."
echo         exit 1
echo     fi
echo else
echo     echo "Skipping end-to-end tests (script not found)"
echo fi
echo 
echo echo "Pre-push checks passed!"
echo exit 0
) > ".git\hooks\pre-push"

echo [OK] Created new pre-push hook
echo.

REM Create a simple pre-commit hook
if not exist ".git\hooks\pre-commit" (
    echo [INFO] Creating pre-commit hook...
    (
    echo #!/bin/sh
    echo # Pre-commit hook for iRis project
    echo echo "Running pre-commit checks..."
    echo # Add your checks here later
    echo exit 0
    ) > ".git\hooks\pre-commit"
    echo [OK] Created pre-commit hook
)

echo ========================================
echo Git Hooks Fixed!
echo ========================================
echo.
echo You can now push normally without errors:
echo   git push
echo.
echo Or continue to bypass hooks when needed:
echo   git push --no-verify
echo.
pause
