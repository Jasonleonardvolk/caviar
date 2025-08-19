Write-Host "`n🔧 TORI Dependency Emergency Fix" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Navigate to the correct directory
Set-Location -Path "C:\Users\jason\Desktop\tori\kha"

# Step 1: Force remove broken scipy directories
Write-Host "`n1️⃣ Removing broken scipy directories..." -ForegroundColor Yellow

$brokenDirs = @(
    ".venv\Lib\site-packages\~cipy",
    ".venv\Lib\site-packages\~cipy-1.16.1.dist-info",
    ".venv\Lib\site-packages\~cipy.libs",
    ".venv\Lib\site-packages\~pds"
)

foreach ($dir in $brokenDirs) {
    if (Test-Path $dir) {
        try {
            # Force remove with elevated permissions
            cmd /c rmdir /s /q $dir 2>$null
            if (Test-Path $dir) {
                # If still exists, try with takeown
                takeown /f $dir /r /d y 2>$null
                icacls $dir /grant "${env:USERNAME}:F" /t /q 2>$null
                Remove-Item -Path $dir -Recurse -Force -ErrorAction Stop
            }
            Write-Host "✅ Removed $dir" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Could not remove $dir - may need admin rights" -ForegroundColor Yellow
        }
    }
}

# Step 2: Quick dependency fix
Write-Host "`n2️⃣ Quick dependency fix..." -ForegroundColor Yellow

# Kill any Python processes that might be holding files
Write-Host "Stopping Python processes..." -ForegroundColor Gray
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force

# Uninstall problematic packages
Write-Host "`nUninstalling problematic packages..." -ForegroundColor Gray
pip uninstall -y numpy scipy scikit-learn transformers sentence-transformers spacy thinc blis 2>$null

# Install compatible versions
Write-Host "`n3️⃣ Installing compatible versions..." -ForegroundColor Yellow

$packages = @(
    "numpy==1.26.4",
    "scipy==1.13.1",
    "scikit-learn==1.5.1", 
    "transformers==4.44.2",
    "sentence-transformers==3.0.1",
    "spacy==3.7.5"
)

$failed = $false
foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Gray
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install $package" -ForegroundColor Red
        $failed = $true
    }
}

# Test the fix
Write-Host "`n4️⃣ Testing imports..." -ForegroundColor Yellow

# Create a temporary test script
$testScript = @'
import sys
sys.path.insert(0, '.')

try:
    import numpy
    print(f'✅ numpy {numpy.__version__}')
    import scipy
    print(f'✅ scipy {scipy.__version__}')
    import sklearn
    print(f'✅ scikit-learn {sklearn.__version__}')
    import transformers
    print(f'✅ transformers {transformers.__version__}')
    import sentence_transformers
    print(f'✅ sentence-transformers {sentence_transformers.__version__}')
    import spacy
    print(f'✅ spacy {spacy.__version__}')
    from ingest_pdf.entropy_prune import entropy_prune
    print('✅ entropy_prune imports successfully!')
    print('\n🎉 All imports successful!')
    exit(0)
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
'@

# Write test script to file
$testScript | Out-File -FilePath "temp_test_imports.py" -Encoding UTF8

# Run the test
python temp_test_imports.py
$testResult = $LASTEXITCODE

# Clean up
Remove-Item "temp_test_imports.py" -ErrorAction SilentlyContinue

if ($testResult -eq 0 -and -not $failed) {
    Write-Host ""
    Write-Host "Everything is fixed! Entropy pruning should work now." -ForegroundColor Green
    Write-Host ""
    Write-Host "Test with:" -ForegroundColor Yellow
    Write-Host "  python test_entropy_state.py" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Some issues remain. Try running:" -ForegroundColor Red
    Write-Host "  python comprehensive_dependency_fix.py" -ForegroundColor Yellow
}
