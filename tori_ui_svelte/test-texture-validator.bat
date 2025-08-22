@echo off
echo =====================================
echo   Testing Enhanced Texture Validator
echo =====================================
echo.
echo New v3.0 Features:
echo   - Texture format allow-list (PNG, JPG, WebP only)
echo   - Alpha channel detection
echo   - Alpha texture limits (2048px, 8MB)
echo   - Normal map limits (2048px)
echo   - Total size validation (50MB)
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo Running enhanced validation...
echo.

node test-enhanced-validator.mjs

echo.
echo =====================================
echo.
echo To test with custom limits:
echo   node tools\assets\validate-manifest.mjs assets\3d\luxury\ASSET_MANIFEST.json --maxAlphaDim=1024
echo.
pause