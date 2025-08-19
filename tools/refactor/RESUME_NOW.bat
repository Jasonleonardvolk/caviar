@echo off
echo Continuing refactoring (skipping the ~30 files already done)...
echo.
cd /d D:\Dev\kha
python tools\refactor\refactor_continue.py --backup-dir "D:\Backups\KhaRefactor_%date:~-4,4%%date:~-10,2%%date:~-7,2%"
pause
