@echo off
echo =====================================
echo   Quick Push with All Fixes
echo =====================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo Step 1: Installing validator dependencies...
call npm install @gltf-transform/core @gltf-transform/functions

echo.
echo Step 2: Committing all changes...
git add .
git commit -m "fix: add missing validator dependencies and push scripts"

echo.
echo Step 3: Pushing to remote...
git push

echo.
echo =====================================
echo   Success! Everything is pushed.
echo =====================================
echo.
pause