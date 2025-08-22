@echo off
color 0A
cls
echo.
echo  ==================================================
echo       ASSET VALIDATOR v2.0 - QUICK TEST
echo  ==================================================
echo.
echo  This will test the enhanced validator that:
echo    - Loads actual GLB/GLTF files
echo    - Counts triangles
echo    - Enforces 100k limit
echo    - Blocks git push on failure
echo.
echo  ==================================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo [1/3] Checking dependencies...
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [X] Node.js not found! Install from https://nodejs.org
    pause
    exit /b 1
)
echo [OK] Node.js found
echo.

echo [2/3] Checking manifest file...
if exist "assets\3d\luxury\ASSET_MANIFEST.json" (
    echo [OK] Manifest found
) else (
    echo [X] Manifest not found at assets\3d\luxury\ASSET_MANIFEST.json
    pause
    exit /b 1
)
echo.

echo [3/3] Running validator with 100k triangle limit...
echo =====================================
echo.

node tools\assets\validate-manifest.mjs assets\3d\luxury\ASSET_MANIFEST.json --maxTris=100000

echo.
if %ERRORLEVEL% equ 0 (
    color 0A
    echo  ==================================================
    echo       VALIDATION PASSED!
    echo  ==================================================
    echo.
    echo  All assets are within the 100k triangle limit.
    echo  Your git push will NOT be blocked.
    echo.
) else (
    color 0C
    echo  ==================================================
    echo       VALIDATION FAILED!
    echo  ==================================================
    echo.
    echo  One or more assets exceed limits.
    echo  Git push would be BLOCKED with these assets.
    echo.
)

echo  Press any key to exit...
pause >nul