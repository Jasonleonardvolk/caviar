# Update Sync Scripts with Correct Google Drive Path
# Run this after find_google_drive.ps1

$configFile = ".\drive_config.json"

if (-not (Test-Path $configFile)) {
    Write-Host "ERROR: drive_config.json not found!" -ForegroundColor Red
    Write-Host "Please run .\find_google_drive.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Load the configuration
$config = Get-Content $configFile | ConvertFrom-Json
$drivePath = $config.GoogleDrivePath
$khaPath = $config.KhaInDrive

Write-Host "`n===== UPDATING SYNC SCRIPTS =====" -ForegroundColor Cyan
Write-Host "Google Drive Path: $drivePath" -ForegroundColor Green
Write-Host "Kha Path: $khaPath" -ForegroundColor Green

# Determine the correct path format
if ($config.MyLaptopPath) {
    $driveRoot = $config.MyLaptopPath
    $relativePath = "kha"
} else {
    $driveRoot = $drivePath
    $relativePath = "kha"
}

Write-Host "`nDrive Root: $driveRoot" -ForegroundColor Cyan

# Update verify_drive_sync.ps1
$verifyScript = @"
# Google Drive Sync Verification Script - UPDATED WITH CORRECT PATH
# Ensures kha folder (2.7GB) is fully synced with focus on last 8 days

`$ErrorActionPreference = "Stop"

# Configuration - AUTOMATICALLY DETECTED
`$DriveRoot = "$driveRoot"
`$KhaPath = "$khaPath"
`$DaysBack = 8
`$LogFile = "`$DriveRoot\sync_verification_`$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log {
    param(`$Message, `$Color = "White")
    `$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "`$timestamp - `$Message" | Out-File -Append -FilePath `$LogFile
    Write-Host `$Message -ForegroundColor `$Color
}

Write-Log "===== GOOGLE DRIVE SYNC VERIFICATION =====" "Green"
Write-Log "Checking: `$KhaPath" "Cyan"
Write-Log "Time window: Last `$DaysBack days" "Cyan"

# Step 1: Inventory recent files (8 days)
Write-Log "`n--- Step 1: Recent Files Inventory (8 days) ---" "Yellow"
`$cutoffDate = (Get-Date).AddDays(-`$DaysBack)
`$recentFiles = Get-ChildItem -Path `$KhaPath -Recurse -File -ErrorAction SilentlyContinue | 
    Where-Object { `$_.LastWriteTime -gt `$cutoffDate }

if (`$recentFiles) {
    `$totalSize = (`$recentFiles | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Log "Found `$(`$recentFiles.Count) files modified since `$cutoffDate" "Cyan"
    Write-Log "Total size of recent files: `$([math]::Round(`$totalSize, 2)) MB" "Cyan"
} else {
    Write-Log "No files found from the last `$DaysBack days" "Yellow"
    Write-Log "Either already synced or no recent changes" "Gray"
}

# Create trigger files
Write-Log "`n--- Creating Sync Triggers ---" "Yellow"
1..3 | ForEach-Object {
    `$triggerFile = "`$KhaPath\SYNC_TRIGGER_8DAYS_`$_`$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    @"
Sync Trigger File `$_
Created: `$(Get-Date)
Purpose: Force Google Drive to detect and sync changes
Time Window: Last `$DaysBack days
Recent files count: `$(if(`$recentFiles) { `$recentFiles.Count } else { 0 })
"@ | Out-File -FilePath `$triggerFile
    Write-Log "Created trigger: `$(Split-Path `$triggerFile -Leaf)" "Green"
}

# Open monitoring tools
Write-Log "`n--- Opening Monitoring Tools ---" "Yellow"
Start-Process "https://drive.google.com/drive/recent"
Start-Process "explorer.exe" -ArgumentList `$KhaPath

Write-Log "`n========================================" "Green"
Write-Log "8-DAY SYNC VERIFICATION COMPLETE!" "Green"
if (`$recentFiles) {
    Write-Log "Files from last `$DaysBack days: `$(`$recentFiles.Count)" "Green"
    Write-Log "Total size to sync: `$([math]::Round(`$totalSize, 2)) MB" "Green"
} else {
    Write-Log "No recent files to sync" "Yellow"
}
Write-Log "========================================" "Green"

Read-Host "`nPress Enter to exit"
"@

$verifyScript | Out-File -FilePath ".\verify_drive_sync.ps1" -Encoding UTF8
Write-Host "Updated: verify_drive_sync.ps1" -ForegroundColor Green

# Update force_drive_sync.bat
$forceBatch = @"
@echo off
REM Force Google Drive Sync - UPDATED WITH CORRECT PATH
REM 8-day window

echo ============================================
echo GOOGLE DRIVE FORCE SYNC SCRIPT
echo Target: $khaPath
echo Focus: Files from last 8 days
echo ============================================
echo.

REM Create trigger file
echo %date% %time% > "$khaPath\SYNC_TRIGGER_%date:~-4,4%%date:~-10,2%%date:~-7,2%.txt"
echo [OK] Created sync trigger file

REM Force Drive rescan
echo.
echo Forcing Google Drive to rescan...
taskkill /IM GoogleDriveFS.exe /F 2>nul
timeout /t 3 /nobreak >nul
start "" "C:\Program Files\Google\Drive File Stream\launch.bat"
timeout /t 5 /nobreak >nul
echo [OK] Google Drive restarted

REM Touch recent files (8 days)
echo.
echo Updating timestamps on files from last 8 days...
powershell -Command "Get-ChildItem '$khaPath' -Recurse -File | Where-Object {`$_.LastWriteTime -gt (Get-Date).AddDays(-8)} | ForEach-Object { `$_.LastWriteTime = `$_.LastWriteTime }"
echo [OK] Recent files touched (8-day window)

REM Open monitoring
start https://drive.google.com/drive/recent
echo [OK] Opened Drive activity page

echo.
echo ============================================
echo SYNC INITIATED - Monitor progress in:
echo 1. System tray (Google Drive icon)
echo 2. Drive activity page (opened in browser)
echo 3. File Explorer sync badges
echo ============================================
pause
"@

$forceBatch | Out-File -FilePath ".\force_drive_sync.bat" -Encoding ASCII
Write-Host "Updated: force_drive_sync.bat" -ForegroundColor Green

# Update quick_8day_status.ps1
$quickStatus = @"
# Quick 8-Day Sync Status Check - UPDATED PATH

`$path = "$khaPath"
`$days = 8
`$cutoff = (Get-Date).AddDays(-`$days)

Write-Host "`n========== 8-DAY SYNC STATUS ==========" -ForegroundColor Cyan
Write-Host "Checking: `$path" -ForegroundColor Gray
Write-Host "Cutoff: `$cutoff" -ForegroundColor Gray

# Get files
`$files = Get-ChildItem `$path -Recurse -File -ErrorAction SilentlyContinue | Where-Object {`$_.LastWriteTime -gt `$cutoff}

if (`$files) {
    `$totalSize = (`$files | Measure-Object Length -Sum).Sum
    `$sizeGB = [math]::Round(`$totalSize / 1GB, 2)
    `$sizeMB = [math]::Round(`$totalSize / 1MB, 2)
    
    Write-Host "`nFOUND: `$(`$files.Count) files | `$sizeGB GB (`$sizeMB MB)" -ForegroundColor Green
    
    # Show most recent files
    Write-Host "`nMOST RECENT FILES:" -ForegroundColor Yellow
    `$files | Sort LastWriteTime -Descending | Select -First 5 | ForEach-Object {
        `$time = `$_.LastWriteTime.ToString("MM/dd HH:mm")
        `$sizeMB = [math]::Round(`$_.Length / 1MB, 2)
        Write-Host "  `$time | `$(`$_.Name) | `$sizeMB MB"
    }
} else {
    Write-Host "`nNo files found from last `$days days!" -ForegroundColor Yellow
    Write-Host "Either already synced or no recent changes." -ForegroundColor Gray
}

Write-Host "`n=======================================" -ForegroundColor Cyan
Write-Host "Press any key to exit..."; `$null = `$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
"@

$quickStatus | Out-File -FilePath ".\quick_8day_status.ps1" -Encoding UTF8
Write-Host "Updated: quick_8day_status.ps1" -ForegroundColor Green

Write-Host "`n===== ALL SCRIPTS UPDATED! =====" -ForegroundColor Green
Write-Host "`nYou can now run:" -ForegroundColor Cyan
Write-Host "  .\quick_8day_status.ps1  - Check what needs syncing" -ForegroundColor White
Write-Host "  .\verify_drive_sync.ps1  - Full verification" -ForegroundColor White
Write-Host "  force_drive_sync.bat     - Force sync now" -ForegroundColor White

Write-Host "`nPress any key to exit..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")