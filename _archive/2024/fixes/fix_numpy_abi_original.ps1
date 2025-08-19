# PowerShell Script to Fix Numpy ABI Compatibility Issues
# Created to resolve: ValueError: numpy.dtype size changed, may indicate binary incompatibility
# Author: Enhanced Assistant
# Date: 2025-08-06

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "NUMPY ABI COMPATIBILITY FIX SCRIPT" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Set the project directory
$projectDir = "C:\Users\jason\Desktop\tori\kha"
Set-Location $projectDir

Write-Host "Working directory: $projectDir" -ForegroundColor Yellow
Write-Host ""

# Step 1: Backup current environment info
Write-Host "[1/7] Creating backup of current environment info..." -ForegroundColor Green
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "$projectDir\backups\env_backup_$timestamp"

if (!(Test-Path "$projectDir\backups")) {
    New-Item -ItemType Directory -Path "$projectDir\backups" -Force | Out-Null
}

New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Save current package list
Write-Host "   - Saving current package list..." -ForegroundColor Gray
poetry show --tree > "$backupDir\packages_tree.txt" 2>&1
poetry show > "$backupDir\packages_list.txt" 2>&1
pip freeze > "$backupDir\pip_freeze.txt" 2>&1

Write-Host "   Backup saved to: $backupDir" -ForegroundColor Gray
Write-Host ""

# Step 2: Uninstall problematic packages
Write-Host "[2/7] Uninstalling problematic packages..." -ForegroundColor Green
Write-Host "   - Uninstalling thinc..." -ForegroundColor Gray
poetry run python -m pip uninstall thinc -y 2>&1 | Out-Null

Write-Host "   - Uninstalling spacy..." -ForegroundColor Gray
poetry run python -m pip uninstall spacy -y 2>&1 | Out-Null

Write-Host "   - Uninstalling numpy..." -ForegroundColor Gray
poetry run python -m pip uninstall numpy -y 2>&1 | Out-Null

Write-Host "   Packages uninstalled successfully" -ForegroundColor Gray
Write-Host ""

# Step 3: Clear Poetry cache
Write-Host "[3/7] Clearing Poetry cache..." -ForegroundColor Green
poetry cache clear pypi --all -n 2>&1 | Out-Null
Write-Host "   Poetry cache cleared" -ForegroundColor Gray
Write-Host ""

# Step 4: Remove virtual environment
Write-Host "[4/7] Removing virtual environment..." -ForegroundColor Green
if (Test-Path ".venv") {
    Write-Host "   - Found .venv directory, removing..." -ForegroundColor Gray
    Remove-Item -Recurse -Force .venv
    Write-Host "   Virtual environment removed" -ForegroundColor Gray
} else {
    Write-Host "   No .venv directory found, skipping..." -ForegroundColor Gray
}

# Also clear Python cache files
Write-Host "   - Clearing Python cache files..." -ForegroundColor Gray
Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force 2>&1 | Out-Null
Get-ChildItem -Path . -Include *.pyc -Recurse -File | Remove-Item -Force 2>&1 | Out-Null
Write-Host "   Python cache cleared" -ForegroundColor Gray
Write-Host ""

# Step 5: Recreate virtual environment and reinstall
Write-Host "[5/7] Recreating virtual environment and reinstalling packages..." -ForegroundColor Green
Write-Host "   This may take several minutes..." -ForegroundColor Yellow

# Check if pyproject.toml exists
if (Test-Path "pyproject.toml") {
    Write-Host "   - Found pyproject.toml, running poetry install..." -ForegroundColor Gray
    poetry install 2>&1 | Out-String | ForEach-Object {
        if ($_ -match "error|failed|Error|Failed") {
            Write-Host $_ -ForegroundColor Red
        } elseif ($_ -match "Installing|Updating|Building") {
            Write-Host $_ -ForegroundColor Gray
        }
    }
    Write-Host "   Poetry installation complete" -ForegroundColor Gray
} else {
    Write-Host "   ERROR: pyproject.toml not found!" -ForegroundColor Red
    Write-Host "   Please ensure you're in the correct project directory" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 6: Verify installations
Write-Host "[6/7] Verifying installations..." -ForegroundColor Green
$testScript = @"
import sys
errors = []
success = []

try:
    import numpy
    success.append(f'numpy {numpy.__version__} imported successfully')
except Exception as e:
    errors.append(f'numpy import failed: {e}')

try:
    import spacy
    success.append(f'spacy {spacy.__version__} imported successfully')
except Exception as e:
    errors.append(f'spacy import failed: {e}')

try:
    import thinc
    success.append(f'thinc {thinc.__version__} imported successfully')
except Exception as e:
    errors.append(f'thinc import failed: {e}')

for s in success:
    print(f'SUCCESS: {s}')
    
for e in errors:
    print(f'ERROR: {e}', file=sys.stderr)
    
sys.exit(len(errors))
"@

$testResult = poetry run python -c $testScript 2>&1
$exitCode = $LASTEXITCODE

foreach ($line in $testResult) {
    if ($line -match "SUCCESS") {
        Write-Host "   $line" -ForegroundColor Green
    } elseif ($line -match "ERROR") {
        Write-Host "   $line" -ForegroundColor Red
    }
}

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "   All packages imported successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "   Some packages failed to import. Please check the errors above." -ForegroundColor Red
}
Write-Host ""

# Step 7: Create logs directory and save results
Write-Host "[7/7] Saving results..." -ForegroundColor Green
if (!(Test-Path "$projectDir\logs")) {
    New-Item -ItemType Directory -Path "$projectDir\logs" -Force | Out-Null
}

$logFile = "$projectDir\logs\numpy_abi_fix_$timestamp.log"
@"
Numpy ABI Fix Results
=====================
Timestamp: $timestamp
Working Directory: $projectDir
Backup Directory: $backupDir

Test Results:
$($testResult -join "`n")

Exit Code: $exitCode
Status: $(if ($exitCode -eq 0) { "SUCCESS" } else { "FAILED" })
"@ | Out-File -FilePath $logFile

Write-Host "   Results saved to: $logFile" -ForegroundColor Gray
Write-Host ""

# Final status
Write-Host "============================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "FIX COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Run your application with:" -ForegroundColor White
    Write-Host "   poetry run python enhanced_launcher.py --api full --require-penrose --enable-hologram --hologram-audio" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "2. To capture logs to file, use:" -ForegroundColor White
    Write-Host "   poetry run python enhanced_launcher.py --api full --require-penrose --enable-hologram --hologram-audio 2>&1 | Tee-Object -FilePath backend_log.txt" -ForegroundColor Cyan
} else {
    Write-Host "FIX ENCOUNTERED ISSUES" -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Please review the error messages above and check:" -ForegroundColor Yellow
    Write-Host "1. Your pyproject.toml file for dependency conflicts" -ForegroundColor White
    Write-Host "2. The log file at: $logFile" -ForegroundColor White
    Write-Host "3. Consider running: poetry update" -ForegroundColor White
}
Write-Host ""