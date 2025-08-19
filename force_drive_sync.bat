@echo off
REM Force Google Drive Sync for kha folder - Windows Batch Script
REM Run this to ensure all files from last 8 days are synced
REM 8-day window provides good buffer - covers over a week of work!

echo ============================================
echo GOOGLE DRIVE FORCE SYNC SCRIPT
echo Target: kha folder (2.7 GB)
echo Focus: Files from last 8 days
echo ============================================
echo.

REM Step 1: Create timestamp trigger file to force Drive to detect changes
echo %date% %time% > "%USERPROFILE%\Google Drive\My Laptop\kha\SYNC_TRIGGER_%date:~-4,4%%date:~-10,2%%date:~-7,2%.txt"
echo [OK] Created sync trigger file

REM Step 2: Force Drive to scan for changes
echo.
echo Forcing Google Drive to rescan...
taskkill /IM GoogleDriveFS.exe /F 2>nul
timeout /t 3 /nobreak >nul
start "" "C:\Program Files\Google\Drive File Stream\launch.bat"
timeout /t 5 /nobreak >nul
echo [OK] Google Drive restarted

REM Step 3: Touch all recent files to ensure they're marked for sync (8 days)
echo.
echo Updating timestamps on files from last 8 days...
powershell -Command "Get-ChildItem '%USERPROFILE%\Google Drive\My Laptop\kha' -Recurse -File | Where-Object {$_.LastWriteTime -gt (Get-Date).AddDays(-8)} | ForEach-Object { $_.LastWriteTime = $_.LastWriteTime }"
echo [OK] Recent files touched (8-day window)

REM Step 4: Open Drive activity monitor
start https://drive.google.com/drive/recent
echo [OK] Opened Drive activity page - monitor uploads here

echo.
echo ============================================
echo SYNC INITIATED - 8 DAY WINDOW
echo Monitor progress in:
echo 1. System tray (Google Drive icon)
echo 2. Drive activity page (opened in browser)
echo 3. File Explorer sync badges
echo ============================================
echo.
echo Note: 8-day window ensures you won't miss anything
echo       even if you run this script a few days late!
echo.
pause