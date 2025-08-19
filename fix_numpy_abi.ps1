# PowerShell Script to Fix NumPy ABI Compatibility Issues with spaCy
# Specifically resolves: numpy.dtype size changed, may indicate binary incompatibility
# Author: Enhanced Assistant  
# Date: 2025-08-06
# Version: 2.0

param(
    [switch]$Quick = $false,      # Quick fix without full reinstall
    [switch]$Force = $false,       # Force operations without prompts
    [switch]$SkipBackup = $false   # Skip backup creation
)

# ANSI color codes
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
    param([string]$Message, [string]$Color = $RESET)
    Write-Host "${Color}${Message}${RESET}"
}

Write-ColorOutput "${BOLD}${CYAN}============================================================${RESET}" $CYAN
Write-ColorOutput "${BOLD}            NUMPY ABI COMPATIBILITY FIX            ${RESET}" $CYAN
Write-ColorOutput "${BOLD}============================================================${RESET}" $CYAN
Write-Host ""

# Set working directory
$projectDir = "C:\Users\jason\Desktop\tori\kha"
Set-Location $projectDir

Write-ColorOutput "Working directory: $projectDir" $YELLOW
Write-ColorOutput "Current time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" $BLUE
Write-Host ""

# Step 1: Environment Check
Write-ColorOutput "[1/7] Checking environment..." $GREEN

# Check Poetry
try {
    $poetryVersion = poetry --version 2>&1
    Write-ColorOutput "   Poetry installed: $poetryVersion" $BLUE
} catch {
    Write-ColorOutput "   ERROR: Poetry not found! Please install Poetry first." $RED
    exit 1
}

# Check current NumPy version
$checkScript = @"
import sys, json
result = {'numpy': None, 'spacy': None, 'error': None}
try:
    import numpy
    result['numpy'] = numpy.__version__
except: pass
try:
    import spacy
    result['spacy'] = spacy.__version__
except Exception as e:
    if 'dtype size changed' in str(e):
        result['error'] = 'ABI_INCOMPATIBILITY'
print(json.dumps(result))
"@

$currentState = poetry run python -c $checkScript 2>&1 | Out-String
try {
    $state = $currentState | ConvertFrom-Json
    if ($state.numpy) {
        Write-ColorOutput "   Current NumPy: $($state.numpy)" $YELLOW
        if ($state.numpy -match "^2\.") {
            Write-ColorOutput "   WARNING: NumPy 2.x detected - incompatible with spaCy!" $RED
        }
    }
    if ($state.error -eq 'ABI_INCOMPATIBILITY') {
        Write-ColorOutput "   ABI incompatibility detected!" $RED
    }
} catch {
    Write-ColorOutput "   Could not determine package versions" $YELLOW
}

Write-Host ""

# Step 2: Backup (unless skipped)
if (-not $SkipBackup) {
    Write-ColorOutput "[2/7] Creating backup..." $GREEN
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = "$projectDir\backups\env_backup_$timestamp"
    
    if (!(Test-Path "$projectDir\backups")) {
        New-Item -ItemType Directory -Path "$projectDir\backups" -Force | Out-Null
    }
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    # Save current state
    poetry show > "$backupDir\packages.txt" 2>&1
    pip freeze > "$backupDir\pip_freeze.txt" 2>&1
    if (Test-Path "pyproject.toml") {
        Copy-Item "pyproject.toml" "$backupDir\pyproject.toml"
    }
    if (Test-Path "poetry.lock") {
        Copy-Item "poetry.lock" "$backupDir\poetry.lock"
    }
    
    Write-ColorOutput "   Backup saved to: $backupDir" $GREEN
} else {
    Write-ColorOutput "[2/7] Skipping backup..." $YELLOW
}

Write-Host ""

# Step 3: Quick fix or full reinstall
if ($Quick) {
    Write-ColorOutput "[3/7] Applying quick fix..." $GREEN
    Write-ColorOutput "   Attempting to fix without full environment rebuild" $BLUE
    
    # Just fix NumPy version
    Write-ColorOutput "   Removing NumPy and spaCy..." $BLUE
    poetry run pip uninstall -y numpy spacy thinc 2>&1 | Out-Null
    
    Write-ColorOutput "   Installing NumPy 1.26.4..." $BLUE
    poetry run pip install numpy==1.26.4 --no-cache-dir 2>&1 | Out-Null
    
    Write-ColorOutput "   Reinstalling thinc and spaCy..." $BLUE
    poetry run pip install thinc spacy --no-cache-dir 2>&1 | Out-Null
    
} else {
    # Full fix process
    Write-ColorOutput "[3/7] Cleaning environment..." $GREEN
    
    # Clear caches
    poetry cache clear pypi --all -n 2>&1 | Out-Null
    Write-ColorOutput "   Poetry cache cleared" $BLUE
    
    # Remove virtual environment
    if (Test-Path ".venv") {
        Write-ColorOutput "   Removing .venv directory..." $BLUE
        Remove-Item -Recurse -Force .venv
    }
    
    # Clear Python caches
    Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory -ErrorAction SilentlyContinue | 
        Remove-Item -Recurse -Force 2>&1 | Out-Null
    Get-ChildItem -Path . -Include *.pyc -Recurse -File -ErrorAction SilentlyContinue | 
        Remove-Item -Force 2>&1 | Out-Null
    Write-ColorOutput "   Environment cleaned" $GREEN
    
    Write-Host ""
    
    # Step 4: Update pyproject.toml
    Write-ColorOutput "[4/7] Updating dependencies..." $GREEN
    
    # Read and update pyproject.toml
    if (Test-Path "pyproject.toml") {
        $content = Get-Content "pyproject.toml" -Raw
        
        # Ensure NumPy 1.26.4
        if ($content -match 'numpy\s*=\s*"[^"]*"') {
            $content = $content -replace 'numpy\s*=\s*"[^"]*"', 'numpy = "^1.26.4"'
            Write-ColorOutput "   Updated NumPy to ^1.26.4" $GREEN
        } else {
            # Add numpy if not present
            if ($content -match '\[tool\.poetry\.dependencies\]') {
                $content = $content -replace '(\[tool\.poetry\.dependencies\][^\[]*)', "`$1numpy = `"^1.26.4`"`n"
                Write-ColorOutput "   Added NumPy ^1.26.4 to dependencies" $GREEN
            }
        }
        
        $content | Set-Content "pyproject.toml" -NoNewline
    }
    
    # Remove poetry.lock for fresh resolution
    if (Test-Path "poetry.lock") {
        Remove-Item "poetry.lock" -Force
        Write-ColorOutput "   Removed poetry.lock for fresh resolution" $BLUE
    }
    
    Write-Host ""
    
    # Step 5: Reinstall
    Write-ColorOutput "[5/7] Installing packages..." $GREEN
    Write-ColorOutput "   This may take several minutes..." $YELLOW
    
    # Create new environment
    poetry env use python 2>&1 | Out-Null
    
    # Install with NumPy constraint
    Write-ColorOutput "   Installing NumPy 1.26.4 first..." $BLUE
    poetry add "numpy@^1.26.4" 2>&1 | Out-String | ForEach-Object {
        if ($_ -match "Installing|Writing") {
            Write-ColorOutput "   $_" $BLUE
        }
    }
    
    # Install all dependencies
    Write-ColorOutput "   Installing all project dependencies..." $BLUE
    poetry install 2>&1 | Out-String | ForEach-Object {
        if ($_ -match "Installing|Package") {
            Write-ColorOutput "   $_" $BLUE
        }
    }
    
    Write-ColorOutput "   Installation complete" $GREEN
}

Write-Host ""

# Step 6: Verification
Write-ColorOutput "[6/7] Verifying installation..." $GREEN

$verifyScript = @"
import sys, json
results = {'success': [], 'failed': []}
packages = ['numpy', 'scipy', 'pandas', 'matplotlib', 'thinc', 'spacy', 'torch', 'numba']

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        results['success'].append(f'{pkg} {ver}')
    except Exception as e:
        results['failed'].append(f'{pkg}: {str(e)[:50]}')

# Test spaCy functionality
try:
    import spacy
    nlp = spacy.blank('en')
    doc = nlp('test')
    results['spacy_works'] = True
except:
    results['spacy_works'] = False

print(json.dumps(results))
"@

$verifyResult = poetry run python -c $verifyScript 2>&1 | Out-String

try {
    $results = $verifyResult | ConvertFrom-Json
    
    foreach ($pkg in $results.success) {
        Write-ColorOutput "   ✓ $pkg" $GREEN
    }
    
    foreach ($pkg in $results.failed) {
        Write-ColorOutput "   ✗ $pkg" $RED
    }
    
    if ($results.spacy_works) {
        Write-ColorOutput "   ✓ spaCy functionality verified!" $GREEN
        $success = $true
    } else {
        Write-ColorOutput "   ✗ spaCy still has issues" $RED
        $success = $false
    }
} catch {
    Write-ColorOutput "   Error during verification" $RED
    $success = $false
}

Write-Host ""

# Step 7: Save results
Write-ColorOutput "[7/7] Saving results..." $GREEN

if (!(Test-Path "$projectDir\logs")) {
    New-Item -ItemType Directory -Path "$projectDir\logs" -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$projectDir\logs\numpy_fix_$timestamp.txt"

@"
NumPy ABI Fix Log
=================
Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Success: $success
Quick Mode: $Quick

Verification Results:
$verifyResult
"@ | Out-File $logFile

Write-ColorOutput "   Log saved to: $logFile" $BLUE
Write-Host ""

# Final status
Write-ColorOutput "${BOLD}${CYAN}============================================================${RESET}" $CYAN

if ($success) {
    Write-ColorOutput "${BOLD}        ✓ FIX COMPLETED SUCCESSFULLY!        ${RESET}" $GREEN
    Write-ColorOutput "${BOLD}${CYAN}============================================================${RESET}" $CYAN
    Write-Host ""
    Write-ColorOutput "Next steps:" $CYAN
    Write-ColorOutput "1. Test your application:" $YELLOW
    Write-ColorOutput "   poetry run python enhanced_launcher.py" $BLUE
    Write-ColorOutput "2. Verify with:" $YELLOW  
    Write-ColorOutput "   poetry run python verify_numpy_abi.py" $BLUE
    Write-ColorOutput "3. Download spaCy models if needed:" $YELLOW
    Write-ColorOutput "   poetry run python -m spacy download en_core_web_sm" $BLUE
} else {
    Write-ColorOutput "${BOLD}        ⚠ FIX INCOMPLETE - MANUAL STEPS NEEDED        ${RESET}" $RED
    Write-ColorOutput "${BOLD}${CYAN}============================================================${RESET}" $CYAN
    Write-Host ""
    Write-ColorOutput "Try these manual steps:" $YELLOW
    Write-ColorOutput "1. Delete .venv folder and poetry.lock" $WHITE
    Write-ColorOutput "2. Run: poetry install" $WHITE
    Write-ColorOutput "3. Run: poetry add numpy@1.26.4" $WHITE
    Write-ColorOutput "4. Run: poetry add spacy thinc" $WHITE
    Write-ColorOutput "" $WHITE
    Write-ColorOutput "Or try the enhanced script:" $YELLOW
    Write-ColorOutput "   .\fix_numpy_abi_enhanced.ps1" $BLUE
}

Write-Host ""