@echo off
color 0A
cls
echo.
echo    ==================================================
echo                 iRis Git Workflow System              
echo    ==================================================
echo.
echo    Welcome! This is your starting point for the
echo    iRis Git workflow and release management.
echo.
echo    ==================================================
echo.
echo    What would you like to do?
echo.
echo    [1] First-time setup (configure Git hooks)
echo    [2] Open interactive workflow menu
echo    [3] Quick status check
echo    [4] Create desktop shortcuts
echo    [5] View command reference
echo    [6] Exit
echo.
set /p choice="Select an option (1-6): "

cd /d D:\Dev\kha\tori_ui_svelte

if "%choice%"=="1" (
    echo.
    echo Running first-time setup...
    echo.
    call setup-git-hooks.bat
    echo.
    echo Setup complete! Press any key to return to menu...
    pause >nul
    START-HERE.bat
)

if "%choice%"=="2" (
    echo.
    echo Opening Git Workflow Menu...
    call GIT-WORKFLOW-MENU.bat
)

if "%choice%"=="3" (
    echo.
    echo Checking project status...
    echo.
    call git-status.bat
    echo.
    pause
    START-HERE.bat
)

if "%choice%"=="4" (
    echo.
    echo Creating desktop shortcuts...
    echo.
    powershell -ExecutionPolicy Bypass -File "Create-GitWorkflowShortcuts.ps1"
    echo.
    pause
    START-HERE.bat
)

if "%choice%"=="5" (
    echo.
    echo Opening command reference...
    start notepad GIT_COMMANDS_QUICKREF.md
    START-HERE.bat
)

if "%choice%"=="6" (
    echo.
    echo Goodbye!
    exit /b
)

echo Invalid choice. Please try again.
timeout /t 2 >nul
START-HERE.bat