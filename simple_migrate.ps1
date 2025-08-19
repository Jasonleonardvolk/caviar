$ErrorActionPreference = "Stop"

# Define paths
$PIGPEN = "C:\Users\jason\Desktop\pigpen"
$TORI = "C:\Users\jason\Desktop\tori\kha"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "TONKA Migration: Pigpen to TORI" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

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
    "use_existing_pdfs.py",
    "prajna\api\prajna_api.py"
)

# Create backup directory
$backupDir = "$TORI\backup_tonka_$TIMESTAMP"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Write-Host "Backup directory: $backupDir" -ForegroundColor Gray

# Copy files
$successCount = 0
$errorCount = 0

Write-Host "`nCopying TONKA files..." -ForegroundColor Yellow

foreach ($file in $tonkaFiles) {
    $sourcePath = Join-Path $PIGPEN $file
    $destPath = Join-Path $TORI $file
    
    if (Test-Path $sourcePath) {
        try {
            # Create destination directory if needed
            $destDir = Split-Path -Parent $destPath
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            
            # Backup if file exists
            if (Test-Path $destPath) {
                $backupFile = Join-Path $backupDir ($file -replace "\\", "_")
                Copy-Item -Path $destPath -Destination $backupFile -Force
            }
            
            # Copy file
            Copy-Item -Path $sourcePath -Destination $destPath -Force
            
            # Fix paths in Python files
            if ($file -like "*.py") {
                $content = Get-Content -Path $destPath -Raw
                $content = $content -replace "C:\\Users\\jason\\Desktop\\pigpen", "C:\\Users\\jason\\Desktop\\tori\\kha"
                $content = $content -replace "C:/Users/jason/Desktop/pigpen", "C:/Users/jason/Desktop/tori/kha"
                Set-Content -Path $destPath -Value $content -NoNewline
            }
            
            Write-Host "  OK: $file" -ForegroundColor Green
            $successCount++
        }
        catch {
            Write-Host "  FAIL: $file - $_" -ForegroundColor Red
            $errorCount++
        }
    }
    else {
        Write-Host "  SKIP: $file (not found)" -ForegroundColor Gray
    }
}

Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "Success: $successCount files" -ForegroundColor Green
Write-Host "Errors: $errorCount files" -ForegroundColor Red
Write-Host "Backup: $backupDir" -ForegroundColor Gray
Write-Host "`nMigration complete!" -ForegroundColor Green
