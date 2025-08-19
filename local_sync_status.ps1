# Simple Local Sync Status - Works without knowing Drive path
# Just checks what needs syncing from THIS folder (last 8 days)

$localPath = Get-Location
$days = 8
$cutoff = (Get-Date).AddDays(-$days)

Write-Host "`n========== LOCAL 8-DAY SYNC STATUS ==========" -ForegroundColor Cyan
Write-Host "Checking: $localPath" -ForegroundColor Gray
Write-Host "Cutoff: $cutoff" -ForegroundColor Gray
Write-Host "This shows what SHOULD sync to Google Drive" -ForegroundColor Yellow

# Get files from last 8 days
$files = Get-ChildItem . -Recurse -File -ErrorAction SilentlyContinue | 
    Where-Object { 
        $_.LastWriteTime -gt $cutoff -and 
        $_.FullName -notmatch "\.git\\" -and
        $_.FullName -notmatch "\.venv\\" -and
        $_.FullName -notmatch "__pycache__" -and
        $_.FullName -notmatch "node_modules\\"
    }

if ($files) {
    $totalSize = ($files | Measure-Object Length -Sum).Sum
    $sizeGB = [math]::Round($totalSize / 1GB, 2)
    $sizeMB = [math]::Round($totalSize / 1MB, 2)
    
    Write-Host "`nFOUND: $($files.Count) files modified in last $days days" -ForegroundColor Green
    Write-Host "Total size: $sizeGB GB ($sizeMB MB)" -ForegroundColor Green
    
    # Group by date
    $byDate = $files | Group-Object {$_.LastWriteTime.Date} | Sort Name -Descending
    
    Write-Host "`nBREAKDOWN BY DAY:" -ForegroundColor Yellow
    foreach ($day in $byDate) {
        $daySize = [math]::Round(($day.Group | Measure-Object Length -Sum).Sum / 1MB, 2)
        $dayName = (Get-Date $day.Name).ToString("MM/dd ddd")
        Write-Host "  $dayName : $($day.Count) files ($daySize MB)"
    }
    
    Write-Host "`nTOP 10 MOST RECENT FILES:" -ForegroundColor Yellow
    $files | Sort LastWriteTime -Descending | Select -First 10 | ForEach-Object {
        $time = $_.LastWriteTime.ToString("MM/dd HH:mm")
        $sizeMB = [math]::Round($_.Length / 1MB, 2)
        $name = if ($_.Name.Length -gt 40) { $_.Name.Substring(0,37) + "..." } else { $_.Name }
        Write-Host "  $time | $name | $sizeMB MB"
    }
    
    Write-Host "`nLARGEST FILES:" -ForegroundColor Yellow  
    $files | Sort Length -Descending | Select -First 5 | ForEach-Object {
        $sizeMB = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  $sizeMB MB | $($_.Name)"
    }
    
    # Check for files that might not sync
    $large = @($files | Where {$_.Length -gt 5GB})
    $temp = @($files | Where {$_.Extension -in '.tmp','.cache','.lock','.log'})
    
    if ($large -or $temp) {
        Write-Host "`nPOTENTIAL SYNC ISSUES:" -ForegroundColor Red
        if ($large) { Write-Host "  - $($large.Count) files over 5GB (Google Drive limit)" -ForegroundColor Red }
        if ($temp) { Write-Host "  - $($temp.Count) temp/cache files (may be ignored)" -ForegroundColor Yellow }
    }
    
    Write-Host "`n===== HOW TO SYNC THESE FILES =====" -ForegroundColor Cyan
    Write-Host "Option 1: Restart Google Drive (most reliable)" -ForegroundColor White
    Write-Host "  1. Right-click Drive icon in system tray" -ForegroundColor Gray
    Write-Host "  2. Click 'Quit'" -ForegroundColor Gray
    Write-Host "  3. Restart Google Drive from Start Menu" -ForegroundColor Gray
    
    Write-Host "`nOption 2: Create a trigger file" -ForegroundColor White
    Write-Host "  Run this command:" -ForegroundColor Gray
    Write-Host '  "SYNC $(Get-Date)" | Out-File ".\FORCE_SYNC_NOW.txt"' -ForegroundColor Green
    
    Write-Host "`nOption 3: Touch the files (updates timestamps)" -ForegroundColor White
    Write-Host "  Run this command:" -ForegroundColor Gray
    Write-Host '  Get-ChildItem . -Recurse | Where {$_.LastWriteTime -gt (Get-Date).AddDays(-8)} | % {$_.LastWriteTime = $_.LastWriteTime}' -ForegroundColor Green
    
} else {
    Write-Host "`nNo files modified in the last $days days!" -ForegroundColor Yellow
    Write-Host "Everything should already be synced." -ForegroundColor Gray
}

Write-Host "`n===== MONITOR SYNC PROGRESS =====" -ForegroundColor Cyan
Write-Host "After triggering sync, check:" -ForegroundColor Yellow
Write-Host "  - System tray: Drive icon shows sync arrows" -ForegroundColor White
Write-Host "  - File Explorer: Green checkmarks on files" -ForegroundColor White  
Write-Host "  - Web: https://drive.google.com/drive/recent" -ForegroundColor White

Write-Host "`n=======================================" -ForegroundColor Cyan
Write-Host "Press any key to exit..."; $null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")