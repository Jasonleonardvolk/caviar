@echo off
echo =====================================
echo   Emergency Push (Bypass Hook)
echo =====================================
echo.
echo This will push WITHOUT running verification.
echo Use only when the hook is broken!
echo.
pause

cd /d D:\Dev\kha\tori_ui_svelte

echo.
echo Pushing to origin without verification...
git push --no-verify -u origin iris/ship-v0.1.0

echo.
echo =====================================
echo Push complete!
echo.
echo Now fix the hook with:
echo   .\fix-git-hook.bat
echo.
pause