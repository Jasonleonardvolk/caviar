# Enhanced PowerShell Script to Fix NumPy ABI Compatibility Issues with spaCy
# Specifically addresses: numpy.dtype size changed error with NumPy 2.x and spaCy
# Author: Enhanced Assistant
# Date: 2025-08-06
# Version: 2.0

param(
    [switch]$Force = $false,
    [switch]$SkipBackup = $false,
    [switch]$Verbose = $false
)

# ANSI color codes for better output
$ESC = [char]27
$RED = "$ESC[91m"
$GREEN = "$ESC[92m"
$YELLOW = "$ESC[93m"
$BLUE = "$ESC[94m"
$MAGENTA = "$ESC[95m"
$CYAN = "$ESC[96m"
$BOLD = "$ESC[1m"
$RESET = "$ESC[0m"

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = $RESET
    )
    Write-Host "${Color}${Message}${RESET}"
}

Write-ColorOutput "${BOLD}${MAGENTA}============================================================${RESET}" $MAGENTA
Write-ColorOutput "${BOLD}            NUMPY ABI COMPATIBILITY FIX SCRIPT            ${RESET}" $MAGENTA
Write-ColorOutput "${BOLD}                     Version 2.0 Enhanced                  ${RESET}" $MAGENTA
Write-ColorOutput "${BOLD}${MAGENTA}============================================================${RESET}" $MAGENTA
Write-Host ""

# Set the project directory
$projectDir = "C:\Users\jason\Desktop\tori\kha"
Set-Location $projectDir

Write-ColorOutput "Working directory: $projectDir" $YELLOW
Write-ColorOutput "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" $BLUE
Write-Host ""

# Step 1: Analyze current environment
Write-ColorOutput "[ANALYSIS] Checking current environment..." $CYAN
Write-ColorOutput "----------------------------------------" $CYAN

# Check if virtual environment exists
$venvExists = Test-Path ".venv"
if ($venvExists) {
    Write-ColorOutput "  Virtual environment found: .venv" $GREEN
} else {
    Write-ColorOutput "  Virtual environment not found" $YELLOW
}

# Check Poetry installation
try {
    $poetryVersion = poetry --version 2>&1
    Write-ColorOutput "  Poetry installed: $poetryVersion" $GREEN
} catch {
    Write-ColorOutput "  Poetry not found! Please install Poetry first." $RED
    exit 1
}

# Check current NumPy version if installed
$numpyCheck = @"
import sys
try:
    import numpy
    print(f'NUMPY_VERSION:{numpy.__version__}')
except:
    print('NUMPY_VERSION:NOT_INSTALLED')
"@

$numpyVersion = poetry run python -c $numpyCheck 2>&1 | Select-String "NUMPY_VERSION:" | ForEach-Object { $_.ToString().Replace("NUMPY_VERSION:", "") }

if ($numpyVersion -and $numpyVersion -ne "NOT_INSTALLED") {
    Write-ColorOutput "  Current NumPy version: $numpyVersion" $YELLOW
    if ($numpyVersion -match "^2\.") {
        Write-ColorOutput "  WARNING: NumPy 2.x detected - this is incompatible with spaCy!" $RED
    }
} else {
    Write-ColorOutput "  NumPy not installed or not accessible" $YELLOW
}

Write-Host ""

# Step 2: Backup current environment (unless skipped)
if (-not $SkipBackup) {
    Write-ColorOutput "[BACKUP] Creating backup of current environment..." $CYAN
    Write-ColorOutput "----------------------------------------" $CYAN
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = "$projectDir\backups\env_backup_$timestamp"
    
    if (!(Test-Path "$projectDir\backups")) {
        New-Item -ItemType Directory -Path "$projectDir\backups" -Force | Out-Null
    }
    
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    # Save current package lists
    Write-ColorOutput "  Saving package information..." $BLUE
    poetry show --tree > "$backupDir\packages_tree.txt" 2>&1
    poetry show > "$backupDir\packages_list.txt" 2>&1
    pip freeze > "$backupDir\pip_freeze.txt" 2>&1
    
    # Copy pyproject.toml and poetry.lock
    if (Test-Path "pyproject.toml") {
        Copy-Item "pyproject.toml" "$backupDir\pyproject.toml"
        Write-ColorOutput "  Backed up pyproject.toml" $GREEN
    }
    if (Test-Path "poetry.lock") {
        Copy-Item "poetry.lock" "$backupDir\poetry.lock"
        Write-ColorOutput "  Backed up poetry.lock" $GREEN
    }
    
    Write-ColorOutput "  Backup saved to: $backupDir" $GREEN
    Write-Host ""
}

# Step 3: Clean existing environment
Write-ColorOutput "[CLEAN] Removing problematic packages and caches..." $CYAN
Write-ColorOutput "----------------------------------------" $CYAN

# Clear Poetry cache
Write-ColorOutput "  Clearing Poetry cache..." $BLUE
poetry cache clear pypi --all -n 2>&1 | Out-Null

# Remove the virtual environment completely
if ($venvExists) {
    Write-ColorOutput "  Removing existing virtual environment..." $BLUE
    Remove-Item -Recurse -Force .venv 2>&1 | Out-Null
    Write-ColorOutput "  Virtual environment removed" $GREEN
}

# Clear Python cache files
Write-ColorOutput "  Clearing Python cache files..." $BLUE
Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force 2>&1 | Out-Null
Get-ChildItem -Path . -Include *.pyc -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force 2>&1 | Out-Null
Write-ColorOutput "  Cache files cleared" $GREEN

# Remove poetry.lock to ensure fresh dependency resolution
if (Test-Path "poetry.lock") {
    Write-ColorOutput "  Removing poetry.lock for fresh dependency resolution..." $BLUE
    Remove-Item "poetry.lock" -Force
    Write-ColorOutput "  poetry.lock removed" $GREEN
}

Write-Host ""

# Step 4: Modify pyproject.toml to use compatible NumPy version
Write-ColorOutput "[CONFIGURE] Updating pyproject.toml for NumPy compatibility..." $CYAN
Write-ColorOutput "----------------------------------------" $CYAN

# Read current pyproject.toml
$pyprojectPath = "$projectDir\pyproject.toml"
if (Test-Path $pyprojectPath) {
    $pyprojectContent = Get-Content $pyprojectPath -Raw
    
    # Check if numpy constraint exists and update it
    if ($pyprojectContent -match 'numpy\s*=\s*"[^"]*"') {
        # Replace existing numpy constraint
        $pyprojectContent = $pyprojectContent -replace 'numpy\s*=\s*"[^"]*"', 'numpy = "^1.26.4"'
        Write-ColorOutput "  Updated existing NumPy constraint to ^1.26.4" $GREEN
    } else {
        Write-ColorOutput "  NumPy not found in dependencies, will be added during installation" $YELLOW
    }
    
    # Save updated pyproject.toml
    $pyprojectContent | Set-Content $pyprojectPath -NoNewline
} else {
    Write-ColorOutput "  ERROR: pyproject.toml not found!" $RED
    exit 1
}

Write-Host ""

# Step 5: Create new virtual environment and install packages
Write-ColorOutput "[INSTALL] Creating new environment and installing packages..." $CYAN
Write-ColorOutput "----------------------------------------" $CYAN
Write-ColorOutput "  This may take several minutes..." $YELLOW
Write-Host ""

# Create new virtual environment
Write-ColorOutput "  Creating new virtual environment..." $BLUE
poetry env use python 2>&1 | Out-Null

# First, explicitly add NumPy 1.26.4
Write-ColorOutput "  Installing NumPy 1.26.4 (compatible version)..." $BLUE
$numpyInstall = poetry add "numpy@^1.26.4" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "  Failed to install NumPy 1.26.4 via Poetry, trying alternative approach..." $YELLOW
    poetry run pip install "numpy==1.26.4" --no-cache-dir 2>&1 | Out-Null
}

# Install all other dependencies
Write-ColorOutput "  Installing all project dependencies..." $BLUE
Write-ColorOutput "  This includes scipy, pandas, matplotlib, thinc, spacy, etc." $BLUE

$installOutput = poetry install 2>&1

# Parse and display installation progress
$installLines = $installOutput -split "`n"
foreach ($line in $installLines) {
    if ($line -match "Installing|Updating|Building") {
        Write-ColorOutput "    $line" $BLUE
    } elseif ($line -match "error|failed|Error|Failed") {
        Write-ColorOutput "    $line" $RED
    } elseif ($Verbose) {
        Write-ColorOutput "    $line" $GRAY
    }
}

Write-ColorOutput "  Package installation complete" $GREEN
Write-Host ""

# Step 6: Verify installations
Write-ColorOutput "[VERIFY] Testing package imports..." $CYAN
Write-ColorOutput "----------------------------------------" $CYAN

$verificationScript = @"
import sys
import json

results = {
    'success': [],
    'errors': [],
    'versions': {}
}

# Test critical packages
packages_to_test = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'thinc',
    'spacy',
    'torch',
    'numba'
]

for package_name in packages_to_test:
    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'unknown')
        results['success'].append(package_name)
        results['versions'][package_name] = version
    except ImportError as e:
        results['errors'].append(f'{package_name}: {str(e)}')
    except Exception as e:
        results['errors'].append(f'{package_name}: Unexpected error - {str(e)}')

# Special check for NumPy ABI compatibility
if 'numpy' in results['success'] and 'spacy' in results['success']:
    try:
        import numpy as np
        import spacy
        # Try to create a spacy object to ensure full compatibility
        nlp = spacy.blank('en')
        results['abi_compatible'] = True
    except Exception as e:
        results['abi_compatible'] = False
        results['abi_error'] = str(e)
else:
    results['abi_compatible'] = False
    results['abi_error'] = 'NumPy or spaCy not available'

print(json.dumps(results, indent=2))
"@

$verificationResult = poetry run python -c $verificationScript 2>&1 | Out-String
try {
    $results = $verificationResult | ConvertFrom-Json
    
    # Display successful imports
    if ($results.success.Count -gt 0) {
        Write-ColorOutput "  Successfully imported packages:" $GREEN
        foreach ($pkg in $results.success) {
            $version = $results.versions.$pkg
            Write-ColorOutput "    $pkg $version" $GREEN
        }
    }
    
    # Display failed imports
    if ($results.errors.Count -gt 0) {
        Write-ColorOutput "  Failed imports:" $RED
        foreach ($error in $results.errors) {
            Write-ColorOutput "    $error" $RED
        }
    }
    
    # Check ABI compatibility
    Write-Host ""
    if ($results.abi_compatible) {
        Write-ColorOutput "  NumPy-spaCy ABI Compatibility: VERIFIED" $GREEN
    } else {
        Write-ColorOutput "  NumPy-spaCy ABI Compatibility: FAILED" $RED
        if ($results.abi_error) {
            Write-ColorOutput "    Error: $($results.abi_error)" $RED
        }
    }
    
    $allSuccess = ($results.errors.Count -eq 0) -and $results.abi_compatible
    
} catch {
    Write-ColorOutput "  Error parsing verification results" $RED
    Write-ColorOutput "  Raw output: $verificationResult" $YELLOW
    $allSuccess = $false
}

Write-Host ""

# Step 7: Save results and logs
Write-ColorOutput "[LOGGING] Saving results..." $CYAN
Write-ColorOutput "----------------------------------------" $CYAN

if (!(Test-Path "$projectDir\logs")) {
    New-Item -ItemType Directory -Path "$projectDir\logs" -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$projectDir\logs\numpy_abi_fix_$timestamp.json"

# Create detailed log
$logData = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    project_directory = $projectDir
    backup_directory = if (-not $SkipBackup) { $backupDir } else { "skipped" }
    numpy_version_before = $numpyVersion
    verification_results = $results
    success = $allSuccess
}

$logData | ConvertTo-Json -Depth 10 | Out-File -FilePath $logFile
Write-ColorOutput "  Results saved to: $logFile" $GREEN
Write-Host ""

# Final status
Write-ColorOutput "${BOLD}${MAGENTA}============================================================${RESET}" $MAGENTA

if ($allSuccess) {
    Write-ColorOutput "${BOLD}           FIX COMPLETED SUCCESSFULLY!           ${RESET}" $GREEN
    Write-ColorOutput "${BOLD}${MAGENTA}============================================================${RESET}" $MAGENTA
    Write-Host ""
    
    Write-ColorOutput "Next steps:" $CYAN
    Write-ColorOutput "1. Test your application:" $YELLOW
    Write-ColorOutput "   poetry run python enhanced_launcher.py --api full --require-penrose --enable-hologram --hologram-audio" $BLUE
    Write-Host ""
    Write-ColorOutput "2. Run your verification script:" $YELLOW
    Write-ColorOutput "   poetry run python verify_numpy_abi.py" $BLUE
    Write-Host ""
    Write-ColorOutput "3. Download spaCy models if needed:" $YELLOW
    Write-ColorOutput "   poetry run python -m spacy download en_core_web_sm" $BLUE
    
} else {
    Write-ColorOutput "${BOLD}         FIX ENCOUNTERED ISSUES - MANUAL INTERVENTION NEEDED         ${RESET}" $RED
    Write-ColorOutput "${BOLD}${MAGENTA}============================================================${RESET}" $MAGENTA
    Write-Host ""
    
    Write-ColorOutput "Troubleshooting steps:" $YELLOW
    Write-ColorOutput "1. Check the log file:" $WHITE
    Write-ColorOutput "   $logFile" $BLUE
    Write-Host ""
    Write-ColorOutput "2. Try manual installation:" $WHITE
    Write-ColorOutput "   poetry remove numpy spacy thinc" $BLUE
    Write-ColorOutput "   poetry add numpy@1.26.4" $BLUE
    Write-ColorOutput "   poetry add thinc spacy" $BLUE
    Write-Host ""
    Write-ColorOutput "3. Consider using conda instead:" $WHITE
    Write-ColorOutput "   conda create -n myenv python=3.11" $BLUE
    Write-ColorOutput "   conda activate myenv" $BLUE
    Write-ColorOutput "   conda install numpy=1.26.4 spacy" $BLUE
}

Write-Host ""
Write-ColorOutput "Script execution completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" $BLUE