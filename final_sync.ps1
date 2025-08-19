$ErrorActionPreference = "Stop"

Write-Host "COMPLETE SYNC: TORI <-> PIGPEN" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

$TORI = "C:\Users\jason\Desktop\tori\kha"
$PIGPEN = "C:\Users\jason\Desktop\pigpen"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# 1. CRITICAL FFT FIX: Pigpen -> TORI
Write-Host "`n1. Copying FFT fix (your concept import fix)..." -ForegroundColor Yellow
Copy-Item "$PIGPEN\frontend\lib\webgpu\fftCompute.ts" "$TORI\frontend\lib\webgpu\fftCompute.ts" -Force
Write-Host "   OK: fftCompute.ts (concept import fix)" -ForegroundColor Green

# 2. ELFIN FILES: TORI -> Pigpen
Write-Host "`n2. Copying ELFIN files to Pigpen..." -ForegroundColor Yellow
if (Test-Path "$TORI\elfin_lsp") {
    Copy-Item "$TORI\elfin_lsp" "$PIGPEN\" -Recurse -Force
    Write-Host "   OK: elfin_lsp directory" -ForegroundColor Green
}

# 3. Any other files modified today in TORI -> Pigpen
Write-Host "`n3. Checking for other TORI changes..." -ForegroundColor Yellow
$todayFiles = Get-ChildItem $TORI -Recurse -File | Where-Object {
    $_.LastWriteTime -gt (Get-Date).Date -and
    $_.FullName -notlike "*backup*" -and
    $_.FullName -notlike "*__pycache__*" -and
    $_.FullName -notlike "*migrate*" -and
    $_.Extension -in ".py", ".js", ".ts", ".jsx", ".tsx", ".json"
}

foreach ($file in $todayFiles) {
    $rel = $file.FullName.Replace("$TORI\", "")
    $dest = "$PIGPEN\$rel"
    
    # Skip if already exists and is identical
    if ((Test-Path $dest) -and (Get-FileHash $file.FullName).Hash -eq (Get-FileHash $dest).Hash) {
        continue
    }
    
    $destDir = Split-Path $dest -Parent
    if (!(Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    
    Copy-Item $file.FullName $dest -Force
    Write-Host "   Copied: $rel" -ForegroundColor Green
}

Write-Host "`n===============================" -ForegroundColor Cyan
Write-Host "SYNC COMPLETE!" -ForegroundColor Green
Write-Host "`nWhat got synced:" -ForegroundColor Yellow
Write-Host "- FFT concept fix: Pigpen -> TORI" -ForegroundColor White
Write-Host "- ELFIN files: TORI -> Pigpen" -ForegroundColor White
Write-Host "- Today's changes: TORI -> Pigpen" -ForegroundColor White
Write-Host "- TONKA files: Already done earlier" -ForegroundColor Gray
