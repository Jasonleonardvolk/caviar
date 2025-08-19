# NUCLEAR FIX - Complete Environment Rebuild with CORRECT NumPy Version
# THIS WILL COMPLETELY REBUILD YOUR ENVIRONMENT FROM SCRATCH
# Author: Enhanced Assistant
# Date: 2025-08-06

Write-Host "`n" -NoNewline
Write-Host "==================================================================" -ForegroundColor Red
Write-Host "     NUCLEAR OPTION - COMPLETE ENVIRONMENT REBUILD" -ForegroundColor Red  
Write-Host "     THIS WILL DELETE AND REBUILD YOUR ENTIRE ENVIRONMENT" -ForegroundColor Red
Write-Host "==================================================================" -ForegroundColor Red
Write-Host ""

Write-Host "CRITICAL: You are trying to install NumPy 2.2.6 - THIS IS WRONG!" -ForegroundColor Red
Write-Host "SpaCy is INCOMPATIBLE with NumPy 2.x!" -ForegroundColor Red
Write-Host "You MUST use NumPy 1.26.4 or lower!" -ForegroundColor Yellow
Write-Host ""

$response = Read-Host "Do you want to proceed with the nuclear fix? (type 'YES' to continue)"
if ($response -ne 'YES') {
    Write-Host "Aborted." -ForegroundColor Yellow
    exit
}

$projectDir = "C:\Users\jason\Desktop\tori\kha"
Set-Location $projectDir

Write-Host "`nStarting nuclear fix at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "Working directory: $projectDir" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill any Python processes
Write-Host "[1/10] Killing Python processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force -ErrorAction SilentlyContinue

# Step 2: Complete backup
Write-Host "[2/10] Creating complete backup..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "$projectDir\nuclear_backup_$timestamp"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Backup everything important
if (Test-Path "pyproject.toml") {
    Copy-Item "pyproject.toml" "$backupDir\pyproject.toml"
}
if (Test-Path "poetry.lock") {
    Copy-Item "poetry.lock" "$backupDir\poetry.lock"
}
if (Test-Path ".env") {
    Copy-Item ".env" "$backupDir\.env"
}

# Save current package list
pip freeze > "$backupDir\pip_freeze.txt" 2>&1
poetry show > "$backupDir\poetry_packages.txt" 2>&1

Write-Host "   Backup saved to: $backupDir" -ForegroundColor Green

# Step 3: Complete environment destruction
Write-Host "[3/10] Destroying current environment..." -ForegroundColor Yellow

# Remove .venv completely
if (Test-Path ".venv") {
    Write-Host "   Removing .venv directory..." -ForegroundColor Red
    cmd /c "rmdir /S /Q .venv" 2>&1 | Out-Null
    Start-Sleep -Seconds 2
    
    # Force remove if still exists
    if (Test-Path ".venv") {
        Remove-Item -Path ".venv" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Clear ALL Python caches
Write-Host "   Clearing all Python caches..." -ForegroundColor Red
Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include *.pyc,*.pyo,*.pyd -Recurse -File | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include .pytest_cache -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Clear pip cache
Write-Host "   Clearing pip cache..." -ForegroundColor Red
pip cache purge 2>&1 | Out-Null

# Clear Poetry cache
Write-Host "   Clearing Poetry cache..." -ForegroundColor Red
poetry cache clear pypi --all -n 2>&1 | Out-Null

# Remove poetry.lock
if (Test-Path "poetry.lock") {
    Write-Host "   Removing poetry.lock..." -ForegroundColor Red
    Remove-Item "poetry.lock" -Force
}

# Step 4: Clean pyproject.toml
Write-Host "[4/10] Fixing pyproject.toml..." -ForegroundColor Yellow

$pyprojectPath = "pyproject.toml"
if (Test-Path $pyprojectPath) {
    $content = Get-Content $pyprojectPath -Raw
    
    # Force NumPy 1.26.4
    if ($content -match 'numpy\s*=\s*"[^"]*"') {
        $content = $content -replace 'numpy\s*=\s*"[^"]*"', 'numpy = "~1.26.4"'
        Write-Host "   Set NumPy to ~1.26.4 (exact version)" -ForegroundColor Green
    } else {
        # Add numpy if not present
        if ($content -match '\[tool\.poetry\.dependencies\]') {
            $content = $content -replace '(\[tool\.poetry\.dependencies\][^\[]*)python\s*=\s*"[^"]*"', "`$0`nnumpy = `"~1.26.4`""
            Write-Host "   Added NumPy ~1.26.4 to dependencies" -ForegroundColor Green
        }
    }
    
    # Ensure spacy doesn't have version constraints that conflict
    if ($content -match 'spacy\s*=\s*"[^"]*"') {
        $content = $content -replace 'spacy\s*=\s*"[^"]*"', 'spacy = "^3.7.5"'
        Write-Host "   Set spaCy to ^3.7.5" -ForegroundColor Green
    }
    
    # Ensure thinc doesn't have version constraints that conflict  
    if ($content -match 'thinc\s*=\s*"[^"]*"') {
        $content = $content -replace 'thinc\s*=\s*"[^"]*"', 'thinc = "^8.2.4"'
        Write-Host "   Set thinc to ^8.2.4" -ForegroundColor Green
    }
    
    $content | Set-Content $pyprojectPath -NoNewline
}

# Step 5: Create fresh environment
Write-Host "[5/10] Creating fresh Python environment..." -ForegroundColor Yellow

# Find Python 3.11
$pythonPath = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $pythonPath) {
    Write-Host "   ERROR: Python not found in PATH!" -ForegroundColor Red
    exit 1
}

Write-Host "   Using Python: $pythonPath" -ForegroundColor Cyan

# Create new venv directly
Write-Host "   Creating new virtual environment..." -ForegroundColor Yellow
& python -m venv .venv

# Activate it
$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "   ERROR: Failed to create virtual environment!" -ForegroundColor Red
    exit 1
}

Write-Host "   Virtual environment created successfully" -ForegroundColor Green

# Step 6: Upgrade pip and install Poetry in venv
Write-Host "[6/10] Setting up pip and build tools..." -ForegroundColor Yellow

& $venvPython -m pip install --upgrade pip setuptools wheel 2>&1 | Out-Null
Write-Host "   Pip upgraded" -ForegroundColor Green

# Step 7: Install NumPy FIRST with exact version
Write-Host "[7/10] Installing NumPy 1.26.4 (CORRECT VERSION)..." -ForegroundColor Yellow

& $venvPip install numpy==1.26.4 --no-cache-dir 2>&1 | Out-String | ForEach-Object {
    if ($_ -match "Successfully installed") {
        Write-Host "   $_" -ForegroundColor Green
    }
}

# Verify NumPy version
$numpyCheck = & $venvPython -c "import numpy; print(f'NumPy version installed: {numpy.__version__}')" 2>&1
Write-Host "   $numpyCheck" -ForegroundColor Cyan

if ($numpyCheck -notmatch "1\.26\.4") {
    Write-Host "   ERROR: Wrong NumPy version installed!" -ForegroundColor Red
    exit 1
}

# Step 8: Install other critical packages in order
Write-Host "[8/10] Installing other packages in dependency order..." -ForegroundColor Yellow

$packages = @(
    "scipy",
    "pandas", 
    "matplotlib",
    "scikit-learn",
    "numba",
    "Cython",
    "cymem",
    "murmurhash",
    "preshed",
    "thinc==8.2.4",
    "spacy==3.7.5"
)

foreach ($pkg in $packages) {
    Write-Host "   Installing $pkg..." -ForegroundColor Cyan
    & $venvPip install $pkg --no-cache-dir 2>&1 | Out-String | ForEach-Object {
        if ($_ -match "Successfully installed") {
            Write-Host "     Success" -ForegroundColor Green
        } elseif ($_ -match "Requirement already satisfied") {
            Write-Host "     Already installed" -ForegroundColor Yellow
        }
    }
}

# Step 9: Configure Poetry to use our venv
Write-Host "[9/10] Configuring Poetry..." -ForegroundColor Yellow

# Tell Poetry to use our venv
poetry env use $venvPython 2>&1 | Out-Null

# Install remaining dependencies via Poetry (without touching numpy/spacy/thinc)
Write-Host "   Installing remaining Poetry dependencies..." -ForegroundColor Cyan
poetry install --no-root 2>&1 | Out-String | ForEach-Object {
    if ($_ -match "Installing|Package") {
        Write-Host "   $_" -ForegroundColor Cyan
    }
}

# Step 10: Final verification
Write-Host "[10/10] Final verification..." -ForegroundColor Yellow

$testScript = @"
import sys
print("Python:", sys.version)
print("-" * 50)

success = True
results = []

# Test imports
packages = [
    ('numpy', '1.26.4'),
    ('scipy', None),
    ('pandas', None),
    ('matplotlib', None),
    ('thinc', '8.2.4'),
    ('spacy', '3.7.5'),
]

for pkg_name, expected_version in packages:
    try:
        pkg = __import__(pkg_name)
        version = getattr(pkg, '__version__', 'unknown')
        
        if expected_version and version != expected_version:
            results.append(f"✗ {pkg_name} {version} (expected {expected_version})")
            success = False
        else:
            results.append(f"✓ {pkg_name} {version}")
    except Exception as e:
        results.append(f"✗ {pkg_name}: {str(e)[:50]}")
        success = False

for r in results:
    print(r)

print("-" * 50)

# Test spaCy functionality
try:
    import spacy
    nlp = spacy.blank('en')
    doc = nlp("This is a test.")
    print("✓ spaCy functionality test PASSED")
except Exception as e:
    print(f"✗ spaCy functionality test FAILED: {e}")
    success = False

print("-" * 50)
if success:
    print("SUCCESS: All tests passed!")
else:
    print("FAILURE: Some tests failed")
    
sys.exit(0 if success else 1)
"@

Write-Host ""
$testResult = & $venvPython -c $testScript 2>&1
$exitCode = $LASTEXITCODE

foreach ($line in $testResult) {
    if ($line -match "✓") {
        Write-Host $line -ForegroundColor Green
    } elseif ($line -match "✗") {
        Write-Host $line -ForegroundColor Red
    } else {
        Write-Host $line -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan

if ($exitCode -eq 0) {
    Write-Host "     NUCLEAR FIX COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Your environment has been completely rebuilt with:" -ForegroundColor Green
    Write-Host "  - NumPy 1.26.4 (CORRECT VERSION)" -ForegroundColor Green
    Write-Host "  - spaCy 3.7.5 (compatible)" -ForegroundColor Green
    Write-Host "  - thinc 8.2.4 (compatible)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Activate your environment:" -ForegroundColor Cyan
    Write-Host "   .\.venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Test your application:" -ForegroundColor Cyan
    Write-Host "   python enhanced_launcher.py" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Download spaCy models:" -ForegroundColor Cyan
    Write-Host "   python -m spacy download en_core_web_sm" -ForegroundColor White
} else {
    Write-Host "     NUCLEAR FIX ENCOUNTERED ISSUES" -ForegroundColor Red
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Your backup is at: $backupDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Try manual conda installation:" -ForegroundColor Yellow
    Write-Host "  conda create -n myenv python=3.11" -ForegroundColor White
    Write-Host "  conda activate myenv" -ForegroundColor White
    Write-Host "  conda install numpy=1.26.4 scipy pandas matplotlib" -ForegroundColor White
    Write-Host "  conda install -c conda-forge spacy" -ForegroundColor White
}

Write-Host ""
Write-Host "Script completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan