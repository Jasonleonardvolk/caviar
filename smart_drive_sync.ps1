# Smart Google Drive Sync for kha folder
# Uses robocopy for one-way sync (better than symlinks for Google Drive)

$ErrorActionPreference = "Stop"

Write-Host "`n===== GOOGLE DRIVE SYNC SOLUTION =====" -ForegroundColor Cyan
Write-Host "Since G:\ doesn't support symbolic links," -ForegroundColor Yellow
Write-Host "we'll use robocopy for smart syncing!" -ForegroundColor Yellow

$source = "C:\Users\jason\Desktop\tori\kha"
$destination = "G:\My Drive\kha"
$days = 8

Write-Host "`nPaths:" -ForegroundColor Gray
Write-Host "  Source: $source" -ForegroundColor White
Write-Host "  Destination: $destination" -ForegroundColor White
Write-Host "  Sync window: Last $days days" -ForegroundColor White

# Check if source exists
if (-not (Test-Path $source)) {
    Write-Host "ERROR: Source folder not found!" -ForegroundColor Red
    exit 1
}

# Check if Google Drive is available
if (-not (Test-Path "G:\My Drive")) {
    Write-Host "ERROR: Google Drive not found at G:\My Drive" -ForegroundColor Red
    exit 1
}

Write-Host "`n===== SYNC OPTIONS =====" -ForegroundColor Cyan
Write-Host "1. MIRROR sync (makes Drive exactly match local)" -ForegroundColor White
Write-Host "   - Copies new/changed files" -ForegroundColor Gray
Write-Host "   - Deletes files not in source" -ForegroundColor Gray
Write-Host "`n2. UPDATE sync (only adds/updates, never deletes)" -ForegroundColor White
Write-Host "   - Copies new/changed files" -ForegroundColor Gray
Write-Host "   - Keeps extra files in Drive" -ForegroundColor Gray
Write-Host "`n3. RECENT sync (only files from last $days days)" -ForegroundColor White
Write-Host "   - Fast, only syncs recent work" -ForegroundColor Gray
Write-Host "`n4. CHECK what needs syncing (dry run)" -ForegroundColor White
Write-Host "   - Shows what would be copied" -ForegroundColor Gray

$choice = Read-Host "`nEnter choice (1-4)"

# Build robocopy command based on choice
$baseCmd = "robocopy `"$source`" `"$destination`""
$excludeDirs = "/XD .git .venv __pycache__ node_modules .tmp.driveupload .tmp.drivedownload"
$excludeFiles = "/XF *.tmp *.cache *.lock *.log"

switch ($choice) {
    "1" {
        # Mirror sync
        Write-Host "`n===== MIRROR SYNC =====" -ForegroundColor Yellow
        $robocopyCmd = "$baseCmd /MIR $excludeDirs $excludeFiles /R:2 /W:5"
        $description = "This will make G:\My Drive\kha exactly match your local folder"
    }
    "2" {
        # Update sync
        Write-Host "`n===== UPDATE SYNC =====" -ForegroundColor Yellow
        $robocopyCmd = "$baseCmd /E $excludeDirs $excludeFiles /R:2 /W:5"
        $description = "This will copy new/changed files without deleting anything"
    }
    "3" {
        # Recent files only (8 days)
        Write-Host "`n===== RECENT FILES SYNC (8 days) =====" -ForegroundColor Yellow
        $maxAge = 8
        $robocopyCmd = "$baseCmd /E /MAXAGE:$maxAge $excludeDirs $excludeFiles /R:2 /W:5"
        $description = "This will only sync files modified in the last $days days"
    }
    "4" {
        # Dry run
        Write-Host "`n===== DRY RUN (Preview) =====" -ForegroundColor Yellow
        $robocopyCmd = "$baseCmd /E /L $excludeDirs $excludeFiles"
        $description = "This will show what would be copied WITHOUT actually copying"
    }
    default {
        Write-Host "Invalid choice!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n$description" -ForegroundColor Cyan
Write-Host "`nCommand to run:" -ForegroundColor Gray
Write-Host "$robocopyCmd" -ForegroundColor White

Write-Host "`nStarting sync..." -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Gray

# Execute robocopy
Invoke-Expression $robocopyCmd

$exitCode = $LASTEXITCODE

# Interpret robocopy exit codes
Write-Host "`n" + "=" * 60 -ForegroundColor Gray
switch ($exitCode) {
    0 { Write-Host "No files were copied. No changes detected." -ForegroundColor Green }
    1 { Write-Host "SUCCESS! Files were copied successfully." -ForegroundColor Green }
    2 { Write-Host "Extra files/dirs detected (normal for update sync)." -ForegroundColor Yellow }
    3 { Write-Host "Files copied + extras detected." -ForegroundColor Green }
    4 { Write-Host "Some mismatched files/dirs detected." -ForegroundColor Yellow }
    5 { Write-Host "Files copied + some mismatches." -ForegroundColor Yellow }
    6 { Write-Host "Extras + mismatches detected." -ForegroundColor Yellow }
    7 { Write-Host "Files copied + extras + mismatches." -ForegroundColor Yellow }
    default { 
        if ($exitCode -ge 8) {
            Write-Host "ERROR: Some files could not be copied (exit code: $exitCode)" -ForegroundColor Red
        }
    }
}

# Show summary
if ($choice -ne "4") {
    Write-Host "`n===== SYNC COMPLETE =====" -ForegroundColor Green
    Write-Host "Your files are now in Google Drive at:" -ForegroundColor Cyan
    Write-Host "  $destination" -ForegroundColor White
    
    # Check what was synced
    $filesInDest = Get-ChildItem $destination -Recurse -File -ErrorAction SilentlyContinue | Measure-Object
    Write-Host "`nTotal files in Drive: $($filesInDest.Count)" -ForegroundColor Green
    
    # Recent files
    $recentCutoff = (Get-Date).AddDays(-$days)
    $recentFiles = Get-ChildItem $destination -Recurse -File -ErrorAction SilentlyContinue | 
        Where-Object { $_.LastWriteTime -gt $recentCutoff }
    
    if ($recentFiles) {
        Write-Host "Files from last $days days: $($recentFiles.Count)" -ForegroundColor Green
        $totalSize = ($recentFiles | Measure-Object Length -Sum).Sum / 1MB
        Write-Host "Recent files size: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Green
    }
}

Write-Host "`n===== NEXT STEPS =====" -ForegroundColor Cyan
Write-Host "1. Check Google Drive sync status in system tray" -ForegroundColor White
Write-Host "2. Visit: https://drive.google.com/drive/recent" -ForegroundColor White
Write-Host "3. Look for green checkmarks in File Explorer" -ForegroundColor White

Write-Host "`n===== AUTOMATION TIP =====" -ForegroundColor Yellow
Write-Host "To sync automatically every day, create a scheduled task:" -ForegroundColor White
Write-Host '  schtasks /create /tn "Sync kha to Drive" /tr "powershell -File C:\Users\jason\Desktop\tori\kha\auto_sync.ps1" /sc daily /st 23:00' -ForegroundColor Gray

Write-Host "`nPress any key to exit..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")