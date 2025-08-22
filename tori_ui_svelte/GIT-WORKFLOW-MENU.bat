@echo off
REM =====================================
REM     iRis Git Workflow Menu
REM =====================================

cd /d D:\Dev\kha\tori_ui_svelte

:MENU
cls
echo =====================================
echo      iRis Git Workflow Menu
echo =====================================
echo.
echo Current Directory: %CD%
echo.
echo 1. Setup Git Hooks (One-time)
echo 2. Test Git Workflow System
echo 3. Check Project Status
echo 4. View Git Workflow Guide
echo 5. Run Verification Tests
echo 6. Create Feature Branch
echo 7. Create Internal Tag
echo 8. Ship Release
echo 9. Exit
echo.
set /p choice="Select option (1-9): "

if "%choice%"=="1" goto SETUP
if "%choice%"=="2" goto TEST
if "%choice%"=="3" goto STATUS
if "%choice%"=="4" goto GUIDE
if "%choice%"=="5" goto VERIFY
if "%choice%"=="6" goto BRANCH
if "%choice%"=="7" goto TAG
if "%choice%"=="8" goto RELEASE
if "%choice%"=="9" goto END
goto MENU

:SETUP
echo.
echo Running Git Hooks Setup...
echo.
call setup-git-hooks.bat
pause
goto MENU

:TEST
echo.
echo Testing Git Workflow System...
echo.
call test-git-workflow.bat
pause
goto MENU

:STATUS
echo.
echo Checking Project Status...
echo.
call git-status.bat
pause
goto MENU

:GUIDE
echo.
echo Opening Git Workflow Guide...
start notepad tools\git\GIT_WORKFLOW_GUIDE.md
goto MENU

:VERIFY
echo.
echo Running Verification Tests...
echo.
powershell -ExecutionPolicy Bypass -File "tools\release\verify-setup.ps1"
pause
goto MENU

:BRANCH
echo.
set /p type="Branch type (feat/fix/docs/chore): "
set /p scope="Scope (e.g., api, renderer): "
set /p name="Branch name: "
echo.
powershell -ExecutionPolicy Bypass -Command "& { .\tools\git\Git-Workflow.ps1 branch '%type%' '%scope%' '%name%' -Push }"
pause
goto MENU

:TAG
echo.
set /p version="Version (e.g., 0.1.1): "
set /p message="Tag message: "
echo.
powershell -ExecutionPolicy Bypass -Command "& { .\tools\git\Git-Workflow.ps1 internal '%version%' -Message '%message%' -Push }"
pause
goto MENU

:RELEASE
echo.
set /p version="Release version (e.g., 1.0.0): "
set /p message="Release message: "
echo.
powershell -ExecutionPolicy Bypass -Command "& { .\tools\git\Git-Workflow.ps1 release '%version%' -Message '%message%' -Push }"
pause
goto MENU

:END
echo.
echo Goodbye!
exit /b