Write-Host "`n🔧 Fixing Numpy Binary Compatibility Issues" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Clean up the broken scipy
Write-Host "`n1️⃣ Cleaning up broken scipy installation..." -ForegroundColor Yellow
$brokenScipy = "C:\Users\jason\Desktop\tori\kha\.venv\Lib\site-packages\~cipy"
if (Test-Path $brokenScipy) {
    Remove-Item -Path $brokenScipy -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Removed broken scipy directory" -ForegroundColor Green
}

# Step 2: Uninstall problematic packages
Write-Host "`n2️⃣ Uninstalling problematic packages..." -ForegroundColor Yellow
pip uninstall -y numpy scipy scikit-learn spacy thinc blis transformers sentence-transformers 2>$null

# Step 3: Clear pip cache
Write-Host "`n3️⃣ Clearing pip cache..." -ForegroundColor Yellow
pip cache purge

# Step 4: Reinstall in correct order
Write-Host "`n4️⃣ Reinstalling packages in correct order..." -ForegroundColor Yellow

Write-Host "`nInstalling numpy first..." -ForegroundColor Green
pip install numpy==2.0.0

Write-Host "`nInstalling scipy..." -ForegroundColor Green
pip install scipy

Write-Host "`nInstalling scikit-learn..." -ForegroundColor Green
pip install scikit-learn==1.7.0

Write-Host "`nInstalling transformers..." -ForegroundColor Green
pip install transformers

Write-Host "`nInstalling sentence-transformers..." -ForegroundColor Green
pip install sentence-transformers

Write-Host "`nInstalling spacy and dependencies..." -ForegroundColor Green
pip install spacy

# Step 5: Test imports
Write-Host "`n5️⃣ Testing imports..." -ForegroundColor Yellow
python -c "
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
"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ All packages installed successfully!" -ForegroundColor Green
    Write-Host "`nNow test entropy pruning:" -ForegroundColor Yellow
    Write-Host "  python verify_entropy.py" -ForegroundColor White
} else {
    Write-Host "`n❌ Some imports failed" -ForegroundColor Red
}
