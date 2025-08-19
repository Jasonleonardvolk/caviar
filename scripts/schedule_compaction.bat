@echo off
REM Schedule TORI Compaction Tasks
REM Creates Windows Task Scheduler tasks for midnight and noon compaction

echo Creating TORI Compaction tasks...

REM Get current directory
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%..\python.exe

REM Check if Python exists, otherwise use system Python
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

REM Create Midnight task
schtasks /create /tn "TORI Compaction Midnight" /tr "\"%PYTHON_EXE%\" \"%SCRIPT_DIR%compact_all_meshes.py\"" /sc daily /st 00:00 /f
if %ERRORLEVEL% EQU 0 (
    echo [OK] Created midnight compaction task
) else (
    echo [ERROR] Failed to create midnight task
)

REM Create Noon task
schtasks /create /tn "TORI Compaction Noon" /tr "\"%PYTHON_EXE%\" \"%SCRIPT_DIR%compact_all_meshes.py\"" /sc daily /st 12:00 /f
if %ERRORLEVEL% EQU 0 (
    echo [OK] Created noon compaction task
) else (
    echo [ERROR] Failed to create noon task
)

REM Create on-startup task
schtasks /create /tn "TORI Compaction Startup" /tr "\"%PYTHON_EXE%\" \"%SCRIPT_DIR%compact_all_meshes.py\"" /sc onstart /delay 0005:00 /f
if %ERRORLEVEL% EQU 0 (
    echo [OK] Created startup compaction task (5 min delay)
) else (
    echo [ERROR] Failed to create startup task
)

echo.
echo Tasks created. You can view them with: schtasks /query /tn "TORI Compaction"
echo To run manually: schtasks /run /tn "TORI Compaction Midnight"
echo To delete: schtasks /delete /tn "TORI Compaction Midnight" /f

pause
