@echo off
REM Quick Action Script - Run This Now!
REM This script will complete the sprint preparation

echo =========================================
echo    QUICK ACTION - SPRINT PREPARATION
echo =========================================
echo.
echo This script will:
echo 1. Run the full sprint prep
echo 2. Show you what to do next
echo.

REM Check if we're in the right directory
if not exist "quick_sprint_prep.bat" (
    echo ERROR: Not in the right directory!
    echo Please run: cd C:\Users\jason\Desktop\tori\kha
    pause
    exit /b 1
)

REM Run the main sprint prep
echo Starting sprint preparation...
echo.
call quick_sprint_prep.bat

echo.
echo =========================================
echo    SPRINT PREP COMPLETE!
echo =========================================
echo.
echo FINAL CHECKLIST:
echo.
echo [x] Repository cleaned
echo [x] Tag created: v0.11.0-hotfix
echo [x] CI optimized
echo [x] Changes committed
echo.
echo [ ] TODO: Update README.md badge with your GitHub username/repo
echo [ ] TODO: Push to GitHub:
echo.
echo     git push origin main
echo     git push origin v0.11.0-hotfix
echo.
echo Then you're ready for the Albert sprint!
echo.
pause
