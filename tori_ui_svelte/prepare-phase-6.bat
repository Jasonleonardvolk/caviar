@echo off
echo ========================================
echo Phase 6 Prep: Archive Non-Canonical Files
echo ========================================
echo.

cd /d D:\Dev\kha\tori_ui_svelte

REM Create archive directory if it doesn't exist
set ARCHIVE_PATH=src\lib\components\archive
if not exist "%ARCHIVE_PATH%" (
    echo Creating archive directory...
    mkdir "%ARCHIVE_PATH%"
    echo Archive directory created: %ARCHIVE_PATH%
) else (
    echo Archive directory already exists: %ARCHIVE_PATH%
)

echo.
echo Checking for non-canonical files to archive...
echo.

REM Archive non-canonical files
set ARCHIVED_COUNT=0

REM HolographicDisplay_FIXED.svelte
if exist "src\lib\components\HolographicDisplay_FIXED.svelte" (
    echo Archiving: HolographicDisplay_FIXED.svelte
    move "src\lib\components\HolographicDisplay_FIXED.svelte" "%ARCHIVE_PATH%\" >nul 2>&1
    echo   -^> Moved to archive
    set /a ARCHIVED_COUNT+=1
) else (
    echo Not found ^(skip^): HolographicDisplay_FIXED.svelte
)

REM HolographicDisplay.svelte.broken
if exist "src\lib\components\HolographicDisplay.svelte.broken" (
    echo Archiving: HolographicDisplay.svelte.broken
    move "src\lib\components\HolographicDisplay.svelte.broken" "%ARCHIVE_PATH%\" >nul 2>&1
    echo   -^> Moved to archive
    set /a ARCHIVED_COUNT+=1
) else (
    echo Not found ^(skip^): HolographicDisplay.svelte.broken
)

REM Backup files
if exist "src\lib\components\HolographicDisplay.svelte.backup_20250815_173912" (
    echo Archiving: HolographicDisplay.svelte.backup_20250815_173912
    move "src\lib\components\HolographicDisplay.svelte.backup_20250815_173912" "%ARCHIVE_PATH%\" >nul 2>&1
    echo   -^> Moved to archive
    set /a ARCHIVED_COUNT+=1
) else (
    echo Not found ^(skip^): HolographicDisplay.svelte.backup_20250815_173912
)

if exist "src\lib\components\HolographicDisplay.svelte.backup_20250815_174152" (
    echo Archiving: HolographicDisplay.svelte.backup_20250815_174152
    move "src\lib\components\HolographicDisplay.svelte.backup_20250815_174152" "%ARCHIVE_PATH%\" >nul 2>&1
    echo   -^> Moved to archive
    set /a ARCHIVED_COUNT+=1
) else (
    echo Not found ^(skip^): HolographicDisplay.svelte.backup_20250815_174152
)

echo.
echo ========================================
echo Archive Summary:
echo   Files archived: %ARCHIVED_COUNT%
echo.

REM Check canonical components
echo Canonical Component Status:
echo.

if exist "src\lib\components\HolographicDisplay.svelte" (
    echo [OK] HolographicDisplay.svelte ^(CANONICAL^)
    for %%F in ("src\lib\components\HolographicDisplay.svelte") do echo      Size: %%~zF bytes
) else (
    echo [ERROR] HolographicDisplay.svelte NOT FOUND!
)

if exist "src\lib\components\HolographicDisplayEnhanced.svelte" (
    echo [OK] HolographicDisplayEnhanced.svelte ^(EXPERIMENTAL^)
    for %%F in ("src\lib\components\HolographicDisplayEnhanced.svelte") do echo      Size: %%~zF bytes
) else (
    echo [WARN] HolographicDisplayEnhanced.svelte not found
)

echo.
echo ========================================
echo Phase 6 preparation complete!
echo.
echo Next steps:
echo 1. Review CANONICAL_COMPONENTS_GUIDE.md
echo 2. Run build to verify no broken imports
echo 3. Proceed with Phase 6 implementation
echo.
pause
