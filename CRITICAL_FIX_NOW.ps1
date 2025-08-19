# CRITICAL TORI FIXES - RUN THIS NOW!
Write-Host "CRITICAL TORI FIXES" -ForegroundColor Red
Write-Host "===================" -ForegroundColor Red

# 1. Fix concept database
Write-Host "`nStep 1: Fixing concept database location and names..." -ForegroundColor Yellow
python fix_concept_db_critical.py

# 2. Verify the fix for ingest_image.py
Write-Host "`nStep 2: Verifying ingest_image.py fix..." -ForegroundColor Yellow
$imageFile = "C:\Users\jason\Desktop\tori\kha\ingest_pdf\pipeline\ingest_image.py"
$firstLine = Get-Content $imageFile -TotalCount 1
if ($firstLine -match "from __future__ import annotations") {
    Write-Host "✅ ingest_image.py has been fixed (annotations import added)" -ForegroundColor Green
} else {
    Write-Host "❌ ingest_image.py still needs fixing!" -ForegroundColor Red
}

# 3. Final checks
Write-Host "`nFINAL VERIFICATION:" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan

# Check concept files
$conceptFile = "C:\Users\jason\Desktop\tori\kha\ingest_pdf\data\concept_file_storage.json"
$seedFile = "C:\Users\jason\Desktop\tori\kha\ingest_pdf\data\concept_seed_universal.json"

if (Test-Path $conceptFile) {
    $size = (Get-Item $conceptFile).Length
    Write-Host "✅ concept_file_storage.json exists ($size bytes)" -ForegroundColor Green
    if ($size -lt 100) {
        Write-Host "⚠️  WARNING: File seems empty! You need to restore your concept data!" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ concept_file_storage.json MISSING!" -ForegroundColor Red
}

if (Test-Path $seedFile) {
    Write-Host "✅ concept_seed_universal.json exists" -ForegroundColor Green
} else {
    Write-Host "❌ concept_seed_universal.json MISSING!" -ForegroundColor Red
}

Write-Host "`n🚀 NOW RUN: python enhanced_launcher.py" -ForegroundColor Green -BackgroundColor DarkGreen
Write-Host "`nYou MUST see:" -ForegroundColor Yellow
Write-Host "  ✅ Main concept storage loaded: 2300+ concepts" -ForegroundColor Cyan
Write-Host "  ✅ No NameError for HolographicDisplayAPI" -ForegroundColor Cyan
Write-Host "  ✅ PDFs process with Method: success" -ForegroundColor Cyan
