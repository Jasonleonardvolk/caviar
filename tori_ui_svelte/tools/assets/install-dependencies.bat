@echo off
echo =====================================
echo   Installing Asset Validator Dependencies
echo =====================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo Installing required packages...
call npm install @gltf-transform/core @gltf-transform/functions

echo.
echo Dependencies installed!
echo.
echo Testing validator...
node tools\assets\validate-manifest.mjs assets\3d\luxury\ASSET_MANIFEST.json --maxTris=100000

echo.
echo =====================================
echo   Ready to push!
echo =====================================
echo.
pause