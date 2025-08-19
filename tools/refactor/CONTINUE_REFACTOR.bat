@echo off
REM CONTINUE REFACTORING - Picks up where it left off!

cd /d D:\Dev\kha

echo ========================================
echo CONTINUING PATH REFACTORING
echo ========================================
echo.
echo This will skip files that were already processed
echo and continue with the remaining ~50,000 files.
echo.
echo Press Ctrl+C anytime to stop (you can resume again later).
echo.

REM Continue with the same backup directory structure
set BACKUP_DIR=D:\Backups\KhaRefactor_%date:~-4,4%%date:~-10,2%%date:~-7,2%

echo Backup directory: %BACKUP_DIR%
echo.

python tools\refactor\refactor_continue.py --backup-dir "%BACKUP_DIR%"

echo.
echo ========================================
echo REFACTORING COMPLETE!
echo Check tools\refactor\ for log files
echo ========================================
pause
