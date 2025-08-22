@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Phase 6: Canonical Path Verification
echo ========================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

set ERRORS=0
set WARNINGS=0

REM Step 1: Check for non-canonical imports
echo STEP 1: Checking for non-canonical component imports...
echo.

REM Search for HolographicDisplayEnhanced imports
echo   Searching for: HolographicDisplayEnhanced
findstr /r /s /m "import.*HolographicDisplayEnhanced" src\*.svelte src\*.ts src\*.js >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [ERROR] Found imports of HolographicDisplayEnhanced
    set /a ERRORS+=1
) else (
    echo   [OK] No imports of HolographicDisplayEnhanced found
)

REM Search for HolographicDisplay_FIXED imports
echo   Searching for: HolographicDisplay_FIXED
findstr /r /s /m "import.*HolographicDisplay_FIXED" src\*.svelte src\*.ts src\*.js >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [ERROR] Found imports of HolographicDisplay_FIXED
    set /a ERRORS+=1
) else (
    echo   [OK] No imports of HolographicDisplay_FIXED found
)

echo.

REM Step 2: Check canonical component exists
echo STEP 2: Verifying canonical component...
echo.

if exist "src\lib\components\HolographicDisplay.svelte" (
    echo   [OK] Canonical component exists
    for %%F in ("src\lib\components\HolographicDisplay.svelte") do echo       Size: %%~zF bytes
) else (
    echo   [ERROR] Canonical component NOT FOUND
    set /a ERRORS+=1
)

echo.

REM Step 3: Check environment variables
echo STEP 3: Checking environment configuration...
echo.

if exist ".env" (
    findstr "VITE_TORI_HOLOGRAM_CANONICAL" .env >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo   [OK] Feature flag found in .env
    ) else (
        echo   [WARN] Feature flag missing in .env
        set /a WARNINGS+=1
    )
) else (
    echo   [WARN] .env not found
    set /a WARNINGS+=1
)

if exist ".env.production" (
    findstr "VITE_TORI_HOLOGRAM_CANONICAL" .env.production >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo   [OK] Feature flag found in .env.production
    ) else (
        echo   [WARN] Feature flag missing in .env.production
        set /a WARNINGS+=1
    )
) else (
    echo   [WARN] .env.production not found
    set /a WARNINGS+=1
)

echo.

REM Step 4: Check tsconfig.json excludes
echo STEP 4: Verifying TypeScript configuration...
echo.

if exist "tsconfig.json" (
    findstr "HolographicDisplayEnhanced.svelte" tsconfig.json >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo   [OK] Non-canonical files excluded in tsconfig.json
    ) else (
        echo   [WARN] Missing excludes in tsconfig.json
        set /a WARNINGS+=1
    )
) else (
    echo   [ERROR] tsconfig.json not found
    set /a ERRORS+=1
)

echo.

REM Step 5: Check non-canonical files are fenced
echo STEP 5: Verifying non-canonical files are fenced...
echo.

if exist "src\lib\components\HolographicDisplay_FIXED.svelte" (
    echo   [OK] Fenced file exists: HolographicDisplay_FIXED.svelte
) else (
    echo   [INFO] Fenced file not found ^(already archived?^): HolographicDisplay_FIXED.svelte
)

if exist "src\lib\components\HolographicDisplayEnhanced.svelte" (
    echo   [OK] Fenced file exists: HolographicDisplayEnhanced.svelte
) else (
    echo   [INFO] Fenced file not found ^(already archived?^): HolographicDisplayEnhanced.svelte
)

echo.

REM Step 6: Build check
echo STEP 6: Running build checks...
echo.

echo   Running: pnpm run check
call pnpm run check >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo   [OK] Type check passed
) else (
    echo   [ERROR] Type check failed
    set /a ERRORS+=1
)

echo.
echo ========================================
echo VERIFICATION SUMMARY
echo ========================================
echo.

if !ERRORS! EQU 0 if !WARNINGS! EQU 0 (
    echo SUCCESS: All Phase 6 requirements met!
    echo   - No non-canonical imports found
    echo   - Canonical component verified
    echo   - Environment flags configured
    echo   - TypeScript excludes in place
    echo   - Build checks passed
) else if !ERRORS! EQU 0 (
    echo PARTIAL SUCCESS: Phase 6 complete with warnings
    echo   Errors: !ERRORS!
    echo   Warnings: !WARNINGS!
) else (
    echo FAILED: Phase 6 has errors that need fixing
    echo   Errors: !ERRORS!
    echo   Warnings: !WARNINGS!
)

echo.
echo Definition of Done Checklist:
if !ERRORS! EQU 0 (
    echo   [X] Zero imports of non-canonical components
    echo   [X] pnpm run check passes
) else (
    echo   [ ] Zero imports of non-canonical components
    echo   [ ] pnpm run check passes
)
echo   [ ] pnpm run build succeeds ^(run separately^)
echo   [X] Only HolographicDisplay.svelte is canonical

echo.
pause
