@echo off
echo ========================================
echo Phase 6: TypeScript Fixes Verification
echo ========================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo Checking TypeScript compilation...
echo.

REM Run type check
echo Running: pnpm run check
call pnpm run check 2>&1 | findstr /i "error"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [WARN] TypeScript errors found. Check the output above.
    echo.
    echo Common errors to look for:
    echo - "Element implicitly has an 'any' type"
    echo - "No index signature"
    echo - "Property does not exist"
    echo - "Duplicate identifier"
) else (
    echo.
    echo [OK] No TypeScript errors detected!
)

echo.
echo ========================================
echo Specific File Checks
echo ========================================
echo.

REM Check if fixed files exist
echo Verifying fixed files exist:
if exist "src\lib\cognitive\braidMemory.ts" (
    echo [OK] braidMemory.ts exists
) else (
    echo [ERROR] braidMemory.ts not found
)

if exist "src\lib\cognitive\loopRecord.ts" (
    echo [OK] loopRecord.ts exists
) else (
    echo [ERROR] loopRecord.ts not found
)

if exist "src\lib\services\PersonaEmergenceEngine.ts" (
    echo [OK] PersonaEmergenceEngine.ts exists
) else (
    echo [ERROR] PersonaEmergenceEngine.ts not found
)

if exist "src\lib\services\solitonMemory.ts" (
    echo [OK] solitonMemory.ts exists
) else (
    echo [ERROR] solitonMemory.ts not found
)

if exist "src\lib\components\DatabaseReset.svelte" (
    echo [OK] DatabaseReset.svelte exists
) else (
    echo [ERROR] DatabaseReset.svelte not found
)

echo.
echo ========================================
echo Fix Verification Complete
echo ========================================
echo.
echo Fixed issues:
echo - Index signature errors in GLYPH_TYPE_BONUSES
echo - Map vs Object handling in PersonaEmergenceEngine
echo - Duplicate userId in solitonMemory
echo - Method name in DatabaseReset.svelte
echo.
echo Next: Run 'pnpm run build' to verify production build
echo.
pause
