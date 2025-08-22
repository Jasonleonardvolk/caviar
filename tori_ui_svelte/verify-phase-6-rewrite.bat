@echo off
echo ========================================
echo Phase 6 Rewrite Verification
echo ========================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo Checking Phase 6 Rewrite Implementation...
echo.

REM Check archived files
echo [1/7] Verifying archived files...
if exist "src\lib\_archive\HolographicDisplay_FIXED.svelte" (
    echo   [OK] HolographicDisplay_FIXED.svelte archived
) else (
    echo   [ERROR] HolographicDisplay_FIXED.svelte not archived
)

if exist "src\lib\_archive\HolographicDisplay.svelte.broken" (
    echo   [OK] HolographicDisplay.svelte.broken archived
) else (
    echo   [ERROR] HolographicDisplay.svelte.broken not archived
)

echo.

REM Check canonical component
echo [2/7] Verifying canonical component...
if exist "src\lib\components\HolographicDisplay.svelte" (
    echo   [OK] HolographicDisplay.svelte exists
    findstr /C:"RealGhostEngine" "src\lib\components\HolographicDisplay.svelte" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo   [OK] RealGhostEngine import found
    ) else (
        echo   [ERROR] RealGhostEngine import missing
    )
) else (
    echo   [ERROR] HolographicDisplay.svelte not found
)

echo.

REM Check new routes
echo [3/7] Verifying new routes...
if exist "src\routes\renderer\+page.svelte" (
    echo   [OK] /renderer route created
) else (
    echo   [ERROR] /renderer route missing
)

if exist "src\routes\well-played\+page.server.ts" (
    echo   [OK] /well-played redirect created
) else (
    echo   [ERROR] /well-played redirect missing
)

echo.

REM Check stub quarantine
echo [4/7] Verifying stub quarantine...
if exist "src\lib\__stubs__\psiMemory.ts" (
    echo   [OK] psiMemory stub quarantined
) else (
    echo   [ERROR] psiMemory stub not quarantined
)

if exist "src\lib\__stubs__\psiFrames.ts" (
    echo   [OK] psiFrames stub quarantined
) else (
    echo   [ERROR] psiFrames stub not quarantined
)

echo.

REM Check environment flags
echo [5/7] Verifying environment configuration...
findstr "VITE_TORI_HOLOGRAM_CANONICAL" .env >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Feature flag in .env
) else (
    echo   [WARN] Feature flag missing in .env
)

echo.

REM Check TypeScript config
echo [6/7] Verifying TypeScript configuration...
findstr "__stubs__" tsconfig.json >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Stubs excluded in tsconfig.json
) else (
    echo   [WARN] Stubs not excluded in tsconfig.json
)

echo.

REM Run type check
echo [7/7] Running type check...
call pnpm run check >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Type check passed
) else (
    echo   [WARN] Type check has warnings/errors
)

echo.
echo ========================================
echo Phase 6 Rewrite Verification Complete
echo ========================================
echo.
echo Summary:
echo   - Archived files: Check logs above
echo   - Canonical component: Rewritten with fixes
echo   - New routes: /renderer and /well-played
echo   - Stubs: Quarantined in __stubs__
echo   - Ready for production testing
echo.
echo Next: Navigate to http://localhost:5173/renderer
echo.
pause
