# Quick 8-Day Sync Status Check
# Run this anytime to see what needs syncing from the last 8 days

$path = "$env:USERPROFILE\Google Drive\My Laptop\kha"
$days = 8
$cutoff = (Get-Date).AddDays(-$days)

Write-Host "`n========== 8-DAY SYNC STATUS ==========" -ForegroundColor Cyan
Write-Host "Checking: $path" -ForegroundColor Gray
Write-Host "Cutoff: $cutoff" -ForegroundColor Gray

# Get files
$files = Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue | Where-Object {$_.LastWriteTime -gt $cutoff}

if ($files) {
    $totalSize = ($files | Measure-Object Length -Sum).Sum
    $sizeGB = [math]::Round($totalSize / 1GB, 2)
    $sizeMB = [math]::Round($totalSize / 1MB, 2)
    
    # Group by date
    $byDate = $files | Group-Object {$_.LastWriteTime.Date} | Sort Name -Descending
    
    Write-Host "`nFOUND: $($files.Count) files | $sizeGB GB ($sizeMB MB)" -ForegroundColor Green
    Write-Host "`nBREAKDOWN BY DAY:" -ForegroundColor Yellow
    
    foreach ($day in $byDate) {
        $daySize = [math]::Round(($day.Group | Measure-Object Length -Sum).Sum / 1MB, 2)
        $dayName = (Get-Date $day.Name).ToString("MM/dd ddd")
        Write-Host "  $dayName : $($day.Count) files ($daySize MB)"
    }
    
    Write-Host "`nMOST RECENT FILES:" -ForegroundColor Yellow
    $files | Sort LastWriteTime -Descending | Select -First 5 | ForEach-Object {
        $time = $_.LastWriteTime.ToString("MM/dd HH:mm")
        $sizeMB = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  $time | $($_.Name) | $sizeMB MB"
    }
    
    Write-Host "`nLARGEST FILES:" -ForegroundColor Yellow  
    $files | Sort Length -Descending | Select -First 5 | ForEach-Object {
        $sizeMB = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  $sizeMB MB | $($_.Name)"
    }
    
    # Check for problem files
    $large = @($files | Where {$_.Length -gt 5GB})
    $temp = @($files | Where {$_.Extension -in '.tmp','.cache','.lock'})
    
    if ($large -or $temp) {
        Write-Host "`nWARNINGS:" -ForegroundColor Red
        if ($large) { Write-Host "  - $($large.Count) files over 5GB (won't sync)" -ForegroundColor Red }
        if ($temp) { Write-Host "  - $($temp.Count) temp/lock files (may not sync)" -ForegroundColor Yellow }
    }
    
    Write-Host "`nQUICK ACTIONS:" -ForegroundColor Cyan
    Write-Host "  1. Run: .\force_drive_sync.bat" -ForegroundColor White
    Write-Host "  2. Or restart Drive: Right-click tray icon > Quit > Restart" -ForegroundColor White
    Write-Host "  3. Check: https://drive.google.com/drive/recent" -ForegroundColor White
    
} else {
    Write-Host "`nNo files found from last $days days!" -ForegroundColor Yellow
    Write-Host "Either already synced or no recent changes." -ForegroundColor Gray
}

Write-Host "`n=======================================" -ForegroundColor Cyan
Write-Host "Press any key to exit..."; $null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")