@echo off
echo =====================================
echo   Applying Git Hook Fixes
echo =====================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo Step 1: Fixing line endings...
powershell -ExecutionPolicy Bypass -File "tools\git\Fix-GitHookLineEndings.ps1"
echo.

echo Step 2: Testing hook functionality...
powershell -ExecutionPolicy Bypass -File "tools\git\Test-PrePushHook.ps1"
echo.

echo =====================================
echo   Fix Complete!
echo =====================================
echo.
echo Your pre-push hook should now work correctly.
echo.
echo To test: git push
echo.
echo If you need to skip asset validation:
echo   PowerShell: $env:IRIS_SKIP_ASSETS = "1"
echo   CMD: set IRIS_SKIP_ASSETS=1
echo   (Remember to unset after!)
echo.
pause