# Quick TORI to Pigpen Migration
# Finds files modified in TORI today and copies them to Pigpen

$ErrorActionPreference = "Stop"

# Define paths
$TORI = "C:\Users\jason\Desktop\tori\kha"
$PIGPEN = "C:\Users\jason\Desktop\pigpen"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "🚀 Recent Changes: TORI → Pigpen" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Get files modified in TORI in the last 24 hours
$cutoffTime = (Get-Date).AddHours(-24)

Write-Host "🔍 Finding files modified since: $($cutoffTime.ToString('yyyy-MM-dd HH:mm'))" -ForegroundColor Gray

$recentFiles = Get-ChildItem -Path $TORI -Recurse -File |
    Where-Object { $_.LastWriteTime -gt $cutoffTime } |
    Where-Object { $_.Extension -in ".py", ".js", ".ts", ".json", ".yaml", ".md" } |
    Where-Object { 
        $_.FullName -notlike "*__pycache__*" -and
        $_.FullName -notlike "*.git*" -and
        $_.FullName -notlike "*node_modules*" -and
        $_.FullName -notlike "*backup*" -and
        $_.FullName -notlike "*migrate*" -and
        $_.Name -ne "tori_status.json" -and
        $_.Name -ne "api_port.json"
    } |
    Sort-Object LastWriteTime -Descending

if ($recentFiles.Count -eq 0) {
    Write-Host "❌ No files modified in TORI in the last 24 hours" -ForegroundColor Yellow
    exit
}

Write-Host "`n📋 Found $($recentFiles.Count) recently modified files:" -ForegroundColor Yellow

# Create backup directory
$backupDir = "$PIGPEN\backup_tori_to_pigpen_$TIMESTAMP"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Function to copy with path fixing
function Copy-ToriToPigpen {
    param($ToriFile)
    
    $relativePath = $ToriFile.FullName.Replace("$TORI\", "")
    $destPath = Join-Path $PIGPEN $relativePath
    
    # Show file info
    $timeAgo = [Math]::Round((New-TimeSpan -Start $ToriFile.LastWriteTime -End (Get-Date)).TotalHours, 1)
    Write-Host "`n  📄 $relativePath" -ForegroundColor White
    Write-Host "     Modified: $($ToriFile.LastWriteTime.ToString('HH:mm:ss')) ($timeAgo hours ago)" -ForegroundColor Gray
    
    # Check if file exists in pigpen
    if (Test-Path $destPath) {
        # Backup existing
        $backupPath = Join-Path $backupDir $relativePath
        $backupDir = Split-Path -Parent $backupPath
        if (-not (Test-Path $backupDir)) {
            New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        }
        Copy-Item -Path $destPath -Destination $backupPath -Force
        Write-Host "     Backed up existing file" -ForegroundColor Gray
    }
    
    # Ensure destination directory exists
    $destDir = Split-Path -Parent $destPath
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    
    # Copy the file
    Copy-Item -Path $ToriFile.FullName -Destination $destPath -Force
    
    # Fix paths if it's a code file
    if ($ToriFile.Extension -in ".py", ".js", ".ts") {
        try {
            $content = Get-Content -Path $destPath -Raw
            
            # Replace TORI paths with pigpen paths
            $content = $content -replace 'C:\\Users\\jason\\Desktop\\tori\\kha', 'C:\\Users\\jason\\Desktop\\pigpen'
            $content = $content -replace 'C:/Users/jason/Desktop/tori/kha', 'C:/Users/jason/Desktop/pigpen'
            $content = $content -replace '\$\{TORI_ROOT\}', '${PIGPEN_ROOT}'
            
            Set-Content -Path $destPath -Value $content -NoNewline
            Write-Host "     ✅ Copied and fixed paths" -ForegroundColor Green
        }
        catch {
            Write-Host "     ✅ Copied (path fixing skipped)" -ForegroundColor Green
        }
    }
    else {
        Write-Host "     ✅ Copied" -ForegroundColor Green
    }
}

# Show files and ask for confirmation
Write-Host "`nFiles to migrate:" -ForegroundColor Yellow
foreach ($file in $recentFiles | Select-Object -First 20) {
    $relativePath = $file.FullName.Replace("$TORI\", "")
    $timeAgo = [Math]::Round((New-TimeSpan -Start $file.LastWriteTime -End (Get-Date)).TotalHours, 1)
    Write-Host "  - $relativePath ($timeAgo hrs ago)" -ForegroundColor White
}

if ($recentFiles.Count -gt 20) {
    Write-Host "  ... and $($recentFiles.Count - 20) more files" -ForegroundColor Gray
}

Write-Host "`n"
$response = Read-Host "Copy these files to Pigpen? (Y/N)"

if ($response -ne 'Y' -and $response -ne 'y') {
    Write-Host "❌ Migration cancelled" -ForegroundColor Yellow
    exit
}

# Copy files
$successCount = 0
$errorCount = 0

foreach ($file in $recentFiles) {
    try {
        Copy-ToriToPigpen -ToriFile $file
        $successCount++
    }
    catch {
        Write-Host "     ❌ Error: $_" -ForegroundColor Red
        $errorCount++
    }
}

# Summary
Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "📊 Migration Summary" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "✅ Success: $successCount files" -ForegroundColor Green
Write-Host "❌ Errors: $errorCount files" -ForegroundColor Red
Write-Host "📁 Backup: $backupDir" -ForegroundColor Gray

Write-Host "`n✅ TORI → Pigpen migration complete!" -ForegroundColor Green
