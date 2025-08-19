# Quick TONKA Migration Script
# Migrates all TONKA-related files from Pigpen to TORI

$ErrorActionPreference = "Stop"

# Define paths
$PIGPEN = "C:\Users\jason\Desktop\pigpen"
$TORI = "C:\Users\jason\Desktop\tori\kha"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "🚀 TONKA Migration: Pigpen → TORI" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# TONKA files to migrate
$tonkaFiles = @(
    "api\tonka_api.py",
    "test_tonka_integration.py",
    "bulk_pdf_processor.py",
    "api\prajna_tonka_coordinator.py",
    "test_prajna_tonka_coordination.py",
    "teach_tonka_from_datasets.py",
    "tonka_education.py",
    "tonka_learning_curriculum.py",
    "tonka_pdf_learner.py",
    "tonka_config.json",
    "process_massive_datasets.py",
    "process_massive_datasets_fixed.py",
    "download_massive_datasets.py",
    "smart_dataset_downloader.py",
    "rapid_code_learner.py",
    "simple_pdf_processor.py",
    "use_existing_pdfs.py"
)

# Modified files (need special handling)
$modifiedFiles = @(
    @{
        Path = "prajna\api\prajna_api.py"
        Description = "Contains TONKA integration"
        BackupFirst = $true
    }
)

# Create backup directory
$backupDir = "$TORI\backup_tonka_$TIMESTAMP"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Write-Host "📁 Backup directory: $backupDir" -ForegroundColor Gray

# Function to copy with path fixing
function Copy-WithPathFix {
    param($Source, $Dest)
    
    # Ensure destination directory exists
    $destDir = Split-Path -Parent $Dest
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    
    # Copy the file
    Copy-Item -Path $Source -Destination $Dest -Force
    
    # Fix paths if it's a Python file
    if ($Dest -like "*.py") {
        $content = Get-Content -Path $Dest -Raw
        
        # Replace pigpen paths with TORI paths
        $content = $content -replace 'C:\\Users\\jason\\Desktop\\pigpen', 'C:\\Users\\jason\\Desktop\\tori\\kha'
        $content = $content -replace 'C:/Users/jason/Desktop/pigpen', 'C:/Users/jason/Desktop/tori/kha'
        $content = $content -replace '\$\{PIGPEN_ROOT\}', '${TORI_ROOT}'
        
        Set-Content -Path $Dest -Value $content -NoNewline
    }
}

# Copy TONKA files
Write-Host "`n📦 Copying TONKA files..." -ForegroundColor Yellow
$successCount = 0
$errorCount = 0

foreach ($file in $tonkaFiles) {
    $sourcePath = Join-Path $PIGPEN $file
    $destPath = Join-Path $TORI $file
    
    if (Test-Path $sourcePath) {
        try {
            Copy-WithPathFix -Source $sourcePath -Dest $destPath
            Write-Host "  ✅ $file" -ForegroundColor Green
            $successCount++
        }
        catch {
            Write-Host "  ❌ $file - $_" -ForegroundColor Red
            $errorCount++
        }
    }
    else {
        Write-Host "  ⏭️ $file (not found)" -ForegroundColor Gray
    }
}

# Handle modified files
Write-Host "`n🔧 Handling modified files..." -ForegroundColor Yellow

foreach ($mod in $modifiedFiles) {
    $sourcePath = Join-Path $PIGPEN $mod.Path
    $destPath = Join-Path $TORI $mod.Path
    
    if (Test-Path $sourcePath) {
        # Backup existing file if requested
        if ($mod.BackupFirst -and (Test-Path $destPath)) {
            $backupName = (Split-Path -Leaf $destPath) + ".backup_$TIMESTAMP"
            $backupPath = Join-Path $backupDir $backupName
            Copy-Item -Path $destPath -Destination $backupPath -Force
            Write-Host "  📋 Backed up: $($mod.Path)" -ForegroundColor Gray
        }
        
        # Copy with path fixing
        try {
            Copy-WithPathFix -Source $sourcePath -Dest $destPath
            Write-Host "  ✅ $($mod.Path) - $($mod.Description)" -ForegroundColor Green
            $successCount++
        }
        catch {
            Write-Host "  ❌ $($mod.Path) - $_" -ForegroundColor Red
            $errorCount++
        }
    }
}

# Quick check for any files we might have missed
Write-Host "`n🔍 Checking for additional TONKA files..." -ForegroundColor Yellow
$additionalFiles = Get-ChildItem -Path $PIGPEN -Filter "*tonka*" -Recurse -File | 
    Where-Object { $_.Extension -in ".py", ".json", ".txt" } |
    Where-Object { $_.FullName -notlike "*__pycache__*" } |
    Select-Object -First 10

if ($additionalFiles) {
    Write-Host "  Found additional TONKA-related files:" -ForegroundColor Cyan
    foreach ($file in $additionalFiles) {
        $relativePath = $file.FullName.Replace("$PIGPEN\", "")
        if ($relativePath -notin $tonkaFiles) {
            Write-Host "    ? $relativePath" -ForegroundColor Yellow
        }
    }
}

# Summary
Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "📊 Migration Summary" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "✅ Success: $successCount files" -ForegroundColor Green
Write-Host "❌ Errors: $errorCount files" -ForegroundColor Red
Write-Host "📁 Backup: $backupDir" -ForegroundColor Gray

# Create a quick verification script
$verifyScript = @'
# Verify TONKA is working in TORI
cd C:\Users\jason\Desktop\tori\kha
python test_tonka_integration.py
'@

$verifyPath = "$TORI\verify_tonka.ps1"
Set-Content -Path $verifyPath -Value $verifyScript

Write-Host "`n✅ Migration complete!" -ForegroundColor Green
Write-Host "📝 Created verification script: verify_tonka.ps1" -ForegroundColor Cyan
Write-Host "`n💡 Next steps:" -ForegroundColor Yellow
Write-Host "  1. cd C:\Users\jason\Desktop\tori\kha" -ForegroundColor White
Write-Host "  2. .\verify_tonka.ps1" -ForegroundColor White
Write-Host "  3. python enhanced_launcher.py" -ForegroundColor White
