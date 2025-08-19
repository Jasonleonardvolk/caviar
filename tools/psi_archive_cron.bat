@echo off
REM PsiArchive maintenance jobs for Windows Task Scheduler

REM Daily seal job - Schedule for 23:59
REM Weekly snapshot - Schedule for Sunday 02:00

set SCRIPT_DIR=%~dp0
set TORI_DIR=%SCRIPT_DIR%\..

REM Activate virtual environment if exists
if exist "%TORI_DIR%\venv\Scripts\activate.bat" (
    call "%TORI_DIR%\venv\Scripts\activate.bat"
)

REM Set Python path
set PYTHONPATH=%TORI_DIR%;%PYTHONPATH%

if "%1"=="seal" goto :seal
if "%1"=="snapshot" goto :snapshot
if "%1"=="verify" goto :verify
goto :usage

:seal
echo [%date% %time%] Running daily seal job...
python "%TORI_DIR%\tools\cron_daily_seal.py"
goto :end

:snapshot
echo [%date% %time%] Creating weekly snapshot...
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set SNAPSHOT_DATE=%%c-%%a-%%b)
python "%TORI_DIR%\tools\psi_replay.py" --until "%SNAPSHOT_DATE%T00:00:00" --output-dir "%TORI_DIR%\data\snapshots\%SNAPSHOT_DATE%"

REM Self-healing snapshot rotation
echo [%date% %time%] Cleaning up old snapshots...

REM Keep last 45 days of snapshots
forfiles /p "%TORI_DIR%\data\snapshots" /d -45 /c "cmd /c if @isdir==TRUE rd /s /q @path" 2>nul

REM Check disk usage
for /f "tokens=3" %%a in ('dir "%TORI_DIR%" ^| find "bytes free"') do set FREE_BYTES=%%a
set /a FREE_GB=%FREE_BYTES:~0,-9% 2>nul

if %FREE_GB% LSS 10 (
    echo   WARNING: Less than 10GB free - aggressive cleanup
    REM Emergency cleanup: keep only last 14 days
    forfiles /p "%TORI_DIR%\data\snapshots" /d -14 /c "cmd /c if @isdir==TRUE if not @fname=="*-full" rd /s /q @path" 2>nul
)
goto :end

:verify
echo [%date% %time%] Verifying archive integrity...
python -c "from core.psi_archive_extended import PSI_ARCHIVER; print('Archive verification not yet implemented')"
goto :end

:usage
echo Usage: %0 {seal^|snapshot^|verify}
exit /b 1

:end
echo [%date% %time%] Job completed
