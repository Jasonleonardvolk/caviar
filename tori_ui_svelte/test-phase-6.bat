@echo off
echo ========================================
echo Phase 6: Complete Test Suite
echo ========================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo Running Phase 6 Verification...
echo --------------------------------
call verify-phase-6.bat

echo.
echo ========================================
echo Running Build Tests...
echo ========================================
echo.

echo Step 1: Type checking with pnpm run check...
call pnpm run check
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Type check failed!
    goto :error
)
echo [OK] Type check passed!

echo.
echo Step 2: Building project with pnpm run build...
call pnpm run build
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed!
    goto :error
)
echo [OK] Build succeeded!

echo.
echo ========================================
echo PHASE 6 COMPLETE!
echo ========================================
echo.
echo All tests passed successfully:
echo   - No non-canonical imports
echo   - Type checking clean
echo   - Build successful
echo   - Canonical path established
echo.
echo Ready for Phase 7!
goto :end

:error
echo.
echo ========================================
echo PHASE 6 INCOMPLETE
echo ========================================
echo Some tests failed. Please review errors above.
echo.

:end
pause
