# Complete TORI Setup Script
Write-Host "`n🚀 TORI Complete Setup Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# 1. Fix concept files
Write-Host "`n📁 Step 1: Fixing concept database files..." -ForegroundColor Yellow
python fix_concept_files.py

# 2. Install missing dependencies
Write-Host "`n📦 Step 2: Installing required packages..." -ForegroundColor Yellow
pip install pydub opencv-python-headless

Write-Host "`n📦 Step 3: Installing recommended packages..." -ForegroundColor Yellow
pip install "pymupdf>=1.24"

# 3. Check ffmpeg
Write-Host "`n🎬 Step 4: Checking ffmpeg installation..." -ForegroundColor Yellow
if (Test-Path "C:\ffmpeg\bin\ffmpeg.exe") {
    Write-Host "✅ ffmpeg found!" -ForegroundColor Green
} else {
    Write-Host "⚠️  ffmpeg not found. Running installer..." -ForegroundColor Yellow
    if (Test-Path ".\install_ffmpeg_simple.ps1") {
        .\install_ffmpeg_simple.ps1
    } else {
        Write-Host "❌ ffmpeg installer not found. Please install manually." -ForegroundColor Red
    }
}

# 4. Check for original concept-mesh data
Write-Host "`n🔍 Step 5: Checking for original concept data..." -ForegroundColor Yellow
$originalConceptMesh = "C:\Users\jason\Desktop\tori\kha\concept-mesh"
$targetConceptMesh = "C:\Users\jason\Desktop\tori\kha\concept_mesh"

if (Test-Path $originalConceptMesh) {
    Write-Host "✅ Found original concept-mesh folder!" -ForegroundColor Green
    Write-Host "   Copying contents to new location..." -ForegroundColor Cyan
    Copy-Item -Path "$originalConceptMesh\*" -Destination $targetConceptMesh -Recurse -Force
    Write-Host "✅ Concept data restored!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Original concept-mesh folder not found at: $originalConceptMesh" -ForegroundColor Yellow
    Write-Host "   If you have it elsewhere, copy it to: $targetConceptMesh" -ForegroundColor Yellow
}

# 5. Final verification
Write-Host "`n✅ Setup complete! Verifying installation..." -ForegroundColor Green

Write-Host "`nChecking concept files:" -ForegroundColor Cyan
if (Test-Path "C:\Users\jason\Desktop\tori\kha\ingest_pdf\data\concept_file_storage.json") {
    Write-Host "✅ concept_file_storage.json exists" -ForegroundColor Green
} else {
    Write-Host "❌ concept_file_storage.json missing" -ForegroundColor Red
}

if (Test-Path "C:\Users\jason\Desktop\tori\kha\ingest_pdf\data\concept_seed_universal.json") {
    Write-Host "✅ concept_seed_universal.json exists" -ForegroundColor Green
} else {
    Write-Host "❌ concept_seed_universal.json missing" -ForegroundColor Red
}

Write-Host "`nChecking Python packages:" -ForegroundColor Cyan
try {
    python -c "import cv2; print('✅ OpenCV installed')" 2>$null
} catch {
    Write-Host "❌ OpenCV missing" -ForegroundColor Red
}

try {
    python -c "import pydub; print('✅ pydub installed')" 2>$null
} catch {
    Write-Host "❌ pydub missing" -ForegroundColor Red
}

Write-Host "`n🎉 Ready to launch TORI!" -ForegroundColor Green
Write-Host "Run: python enhanced_launcher.py" -ForegroundColor Yellow
