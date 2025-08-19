# Google Drive Sync Verification Script
# Ensures kha folder (2.7GB) is fully synced with focus on last 8 days
# 8-day window = over a week of work + buffer if running late!

$ErrorActionPreference = "Stop"

# Configuration
$DriveRoot = "$env:USERPROFILE\Google Drive\My Laptop"
$KhaPath = "$DriveRoot\kha"
$DaysBack = 8  # Changed from 3 to 8 for better coverage
$LogFile = "$DriveRoot\sync_verification_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log {
    param($Message, $Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -Append -FilePath $LogFile
    Write-Host $Message -ForegroundColor $Color
}

Write-Log "===== GOOGLE DRIVE SYNC VERIFICATION =====" "Green"
Write-Log "Checking: $KhaPath" "Cyan"
Write-Log "Time window: Last $DaysBack days" "Cyan"

# Step 1: Inventory recent files (8 days)
Write-Log "`n--- Step 1: Recent Files Inventory (8 days) ---" "Yellow"
$cutoffDate = (Get-Date).AddDays(-$DaysBack)
$recentFiles = Get-ChildItem -Path $KhaPath -Recurse -File -ErrorAction SilentlyContinue | 
    Where-Object { $_.LastWriteTime -gt $cutoffDate }

$totalSize = ($recentFiles | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Log "Found $($recentFiles.Count) files modified since $cutoffDate" "Cyan"
Write-Log "Total size of recent files: $([math]::Round($totalSize, 2)) MB" "Cyan"

# Step 2: Check for sync status indicators
Write-Log "`n--- Step 2: Creating File Manifest ---" "Yellow"

# Create manifest with checksums
$manifest = @()
$fileCounter = 0
foreach ($file in $recentFiles) {
    $fileCounter++
    if ($fileCounter % 10 -eq 0) {
        Write-Host "Processing file $fileCounter of $($recentFiles.Count)..." -ForegroundColor Gray
    }
    
    $relativePath = $file.FullName.Replace($KhaPath, "").TrimStart("\")
    $hash = (Get-FileHash -Path $file.FullName -Algorithm MD5).Hash
    $manifest += [PSCustomObject]@{
        Path = $relativePath
        Size = $file.Length
        Modified = $file.LastWriteTime
        Hash = $hash
    }
}

# Save manifest
$manifestPath = "$DriveRoot\recent_files_manifest_8days.csv"
$manifest | Export-Csv -Path $manifestPath -NoTypeInformation
Write-Log "Manifest saved to: $manifestPath" "Green"

# Step 3: Check for problem indicators
Write-Log "`n--- Step 3: Checking for Issues ---" "Yellow"

# Check for files over 5GB (Drive limit)
$largeFiles = $recentFiles | Where-Object { $_.Length -gt 5GB }
if ($largeFiles) {
    Write-Log "WARNING: Found files over 5GB limit:" "Red"
    $largeFiles | ForEach-Object { Write-Log "  - $($_.Name) ($([math]::Round($_.Length/1GB, 2)) GB)" "Red" }
}

# Check for problematic file types
$problemExtensions = @('.tmp', '.lock', '.~lock', '.cache')
$problemFiles = $recentFiles | Where-Object { $problemExtensions -contains $_.Extension }
if ($problemFiles) {
    Write-Log "WARNING: Found temporary/lock files that may not sync:" "Yellow"
    $problemFiles | ForEach-Object { Write-Log "  - $($_.Name)" "Yellow" }
}

# Check for special characters in filenames
$specialCharFiles = $recentFiles | Where-Object { $_.Name -match '[<>:"|?*]' }
if ($specialCharFiles) {
    Write-Log "WARNING: Files with special characters may have sync issues:" "Yellow"
    $specialCharFiles | ForEach-Object { Write-Log "  - $($_.Name)" "Yellow" }
}

# Step 4: Force sync trigger
Write-Log "`n--- Step 4: Triggering Sync ---" "Yellow"

# Create multiple trigger files to ensure Drive notices
1..3 | ForEach-Object {
    $triggerFile = "$KhaPath\SYNC_TRIGGER_8DAYS_$_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    @"
Sync Trigger File $_
Created: $(Get-Date)
Purpose: Force Google Drive to detect and sync changes
Time Window: Last $DaysBack days
Recent files count: $($recentFiles.Count)
Total size: $([math]::Round($totalSize, 2)) MB
Cutoff date: $cutoffDate
"@ | Out-File -FilePath $triggerFile
    Write-Log "Created trigger: $(Split-Path $triggerFile -Leaf)" "Green"
}

# Step 5: Generate detailed report
Write-Log "`n--- Step 5: Generating Sync Report ---" "Yellow"

# Group files by date for better overview
$filesByDate = $recentFiles | Group-Object { $_.LastWriteTime.Date } | Sort-Object Name -Descending

$report = @"

8-DAY SYNC VERIFICATION REPORT
===============================
Generated: $(Get-Date)
Time Window: $cutoffDate to now ($DaysBack days)

SUMMARY:
--------
Total Files: $($recentFiles.Count)
Total Size: $([math]::Round($totalSize, 2)) MB
Oldest File: $(($recentFiles | Sort-Object LastWriteTime | Select-Object -First 1).LastWriteTime)
Newest File: $(($recentFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime)

FILES BY DAY:
-------------
"@

foreach ($dayGroup in $filesByDate) {
    $daySize = ($dayGroup.Group | Measure-Object -Property Length -Sum).Sum / 1MB
    $report += "`n$(Get-Date $dayGroup.Name -Format 'yyyy-MM-dd dddd'): $($dayGroup.Count) files ($([math]::Round($daySize, 2)) MB)"
}

$report += @"

MANUAL VERIFICATION CHECKLIST:
==============================
[  ] Check Google Drive icon in system tray (should show sync activity)
[  ] No error badges on Drive icon
[  ] Files in Explorer show green checkmarks at: $KhaPath
[  ] No "sync pending" icons on recent files
[  ] Drive web shows recent activity: https://drive.google.com/drive/recent
[  ] 'kha' folder shows today's date in Drive web interface

TOP 10 MOST RECENT FILES TO VERIFY:
------------------------------------
"@

# List top 10 most recent files for manual check
$recentFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 10 | ForEach-Object {
    $report += "`n- $($_.Name)"
    $report += "`n  Modified: $('{0:yyyy-MM-dd HH:mm:ss}' -f $_.LastWriteTime)"
    $report += "`n  Size: $([math]::Round($_.Length/1KB, 2)) KB"
}

$report += @"

POTENTIAL ISSUES:
-----------------
Large files (>5GB): $($largeFiles.Count)
Temporary files: $($problemFiles.Count)
Special char files: $($specialCharFiles.Count)

NEXT STEPS IF SYNC IS STUCK:
-----------------------------
1. Right-click Drive icon -> Quit
2. Wait 30 seconds
3. Restart Google Drive
4. Run this script again
5. Check error log at: %LOCALAPPDATA%\Google\DriveFS\Logs

OUTPUT FILES:
-------------
Log: $LogFile
Manifest: $manifestPath

"@

Write-Log $report "Cyan"

# Save report
$reportPath = "$DriveRoot\sync_report_8days_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
$report | Out-File -FilePath $reportPath
Write-Log "Report saved to: $reportPath" "Green"

# Step 6: Open monitoring tools
Write-Log "`n--- Step 6: Opening Monitoring Tools ---" "Yellow"
Start-Process "https://drive.google.com/drive/recent"
Start-Process "explorer.exe" -ArgumentList $KhaPath

Write-Log "`n========================================" "Green"
Write-Log "8-DAY SYNC VERIFICATION COMPLETE!" "Green"
Write-Log "Files from last $DaysBack days: $($recentFiles.Count)" "Green"
Write-Log "Total size to sync: $([math]::Round($totalSize, 2)) MB" "Green"
Write-Log "========================================" "Green"

# Keep console open
Read-Host "`nPress Enter to exit"