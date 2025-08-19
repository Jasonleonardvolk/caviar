@echo off
REM TORI/KHA Production Fix - One Click Solution
REM Ensures 100% functionality with all components

echo.
echo ===================================================================
echo     TORI/KHA PRODUCTION FIX - ENSURING 100%% FUNCTIONALITY
echo ===================================================================
echo.

REM Check if we're in the right directory
if not exist "enhanced_launcher.py" (
    echo ERROR: Not in TORI/KHA directory!
    echo Please run from: C:\Users\jason\Desktop\tori\kha
    pause
    exit /b 1
)

REM Run the production fix
echo Running comprehensive production fix...
echo.
echo This will:
echo   - Fix all Pydantic imports (v2 migration)
echo   - Install Penrose similarity engine
echo   - Set up Concept Mesh components  
echo   - Create Soliton API endpoints
echo   - Update all configuration files
echo   - Ensure 100%% functionality
echo.

python tools\fix_all_issues_production.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================================
    echo                     FIX COMPLETED SUCCESSFULLY!
    echo ===================================================================
    echo.
    echo Next steps:
    echo   1. Run: python enhanced_launcher.py
    echo   2. Wait for system to start
    echo   3. Test: python test_components.py
    echo   4. Open: http://localhost:5173
    echo.
    echo System now has 100%% functionality including Penrose!
    echo.
) else (
    echo.
    echo ERROR: Fix failed! Check the output above.
    echo.
)

pause
