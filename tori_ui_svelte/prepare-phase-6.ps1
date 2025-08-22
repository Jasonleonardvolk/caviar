# Phase 6 Preparation - Archive Non-Canonical Components
# This script archives non-canonical holographic display components

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 6 Prep: Archive Non-Canonical Files" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path "D:\Dev\kha\tori_ui_svelte"

# Create archive directory if it doesn't exist
$archivePath = "src\lib\components\archive"
if (-not (Test-Path $archivePath)) {
    Write-Host "Creating archive directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $archivePath -Force | Out-Null
    Write-Host "Archive directory created: $archivePath" -ForegroundColor Green
} else {
    Write-Host "Archive directory already exists: $archivePath" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Checking for non-canonical files to archive..." -ForegroundColor Yellow
Write-Host ""

# Files to archive (non-canonical)
$filesToArchive = @(
    "src\lib\components\HolographicDisplay_FIXED.svelte",
    "src\lib\components\HolographicDisplay.svelte.broken",
    "src\lib\components\HolographicDisplay.svelte.backup_20250815_173912",
    "src\lib\components\HolographicDisplay.svelte.backup_20250815_174152"
)

$archivedCount = 0
foreach ($file in $filesToArchive) {
    if (Test-Path $file) {
        $fileName = Split-Path $file -Leaf
        $destination = Join-Path $archivePath $fileName
        
        Write-Host "Archiving: $fileName" -ForegroundColor Yellow
        Move-Item -Path $file -Destination $destination -Force
        Write-Host "  -> Moved to archive" -ForegroundColor Green
        $archivedCount++
    } else {
        Write-Host "Not found (skip): $(Split-Path $file -Leaf)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Archive Summary:" -ForegroundColor Cyan
Write-Host "  Files archived: $archivedCount" -ForegroundColor White
Write-Host ""

# Report on canonical components status
Write-Host "Canonical Component Status:" -ForegroundColor Cyan
Write-Host ""

$canonicalFile = "src\lib\components\HolographicDisplay.svelte"
if (Test-Path $canonicalFile) {
    Write-Host "[OK] HolographicDisplay.svelte (CANONICAL)" -ForegroundColor Green
    $fileInfo = Get-Item $canonicalFile
    Write-Host "     Size: $($fileInfo.Length) bytes" -ForegroundColor Gray
    Write-Host "     Modified: $($fileInfo.LastWriteTime)" -ForegroundColor Gray
} else {
    Write-Host "[ERROR] HolographicDisplay.svelte NOT FOUND!" -ForegroundColor Red
}

$experimentalFile = "src\lib\components\HolographicDisplayEnhanced.svelte"
if (Test-Path $experimentalFile) {
    Write-Host "[OK] HolographicDisplayEnhanced.svelte (EXPERIMENTAL)" -ForegroundColor Yellow
    $fileInfo = Get-Item $experimentalFile
    Write-Host "     Size: $($fileInfo.Length) bytes" -ForegroundColor Gray
    Write-Host "     Modified: $($fileInfo.LastWriteTime)" -ForegroundColor Gray
} else {
    Write-Host "[WARN] HolographicDisplayEnhanced.svelte not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 6 preparation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Review CANONICAL_COMPONENTS_GUIDE.md" -ForegroundColor White
Write-Host "2. Run build to verify no broken imports" -ForegroundColor White
Write-Host "3. Proceed with Phase 6 implementation" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
