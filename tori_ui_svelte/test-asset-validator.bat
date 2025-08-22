@echo off
echo =====================================
echo    Asset Manifest Validator Test
echo =====================================
echo.
echo This will validate all 3D assets in the manifest
echo and check triangle counts.
echo.

cd /d D:\Dev\kha\tori_ui_svelte

REM Check if Node.js is installed
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [X] Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

REM Check if manifest exists
if not exist "assets\3d\luxury\ASSET_MANIFEST.json" (
    echo [X] Asset manifest not found!
    echo Expected at: assets\3d\luxury\ASSET_MANIFEST.json
    pause
    exit /b 1
)

REM Run validator with different triangle limits
echo Running validator with 100k triangle limit...
echo.
call pnpm run validate:assets

if %ERRORLEVEL% equ 0 (
    echo.
    echo =====================================
    echo    Validation PASSED!
    echo =====================================
    echo All assets are within limits.
) else (
    echo.
    echo =====================================
    echo    Validation FAILED!
    echo =====================================
    echo Fix the issues above before pushing.
)

echo.
echo You can also test with different limits:
echo   node tools\assets\validate-manifest.mjs assets\3d\luxury\ASSET_MANIFEST.json --maxTris=50000
echo   node tools\assets\validate-manifest.mjs assets\3d\luxury\ASSET_MANIFEST.json --maxTris=200000
echo.
pause