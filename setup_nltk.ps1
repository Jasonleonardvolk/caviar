#!/usr/bin/env pwsh
# Install NLTK and download required data packages for TORI

Write-Host "`n=== Installing NLTK for TORI ===" -ForegroundColor Cyan
Write-Host "This will add NLTK to the project and download required data packages." -ForegroundColor Gray

# Check if we're in a poetry environment
try {
    $poetryCheck = poetry env info 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] Not in a Poetry environment. Run 'poetry shell' first." -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "[ERROR] Poetry not found. Please ensure Poetry is installed." -ForegroundColor Red
    exit 1
}

# Step 1: Install NLTK via poetry
Write-Host "`n[1] Adding NLTK to dependencies..." -ForegroundColor Yellow
try {
    poetry add nltk
    Write-Host "[OK] NLTK added to project dependencies" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to add NLTK: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Download NLTK data
Write-Host "`n[2] Downloading NLTK data packages..." -ForegroundColor Yellow
try {
    # Method 1: Using the Python script
    if (Test-Path "download_nltk_data.py") {
        poetry run python download_nltk_data.py
    } else {
        # Method 2: Direct command
        poetry run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    }
    Write-Host "[OK] NLTK data packages downloaded" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to download NLTK data: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Verify installation
Write-Host "`n[3] Verifying NLTK installation..." -ForegroundColor Yellow
$verifyScript = @'
import nltk
try:
    # Test punkt
    nltk.data.find('tokenizers/punkt')
    print("[OK] punkt tokenizer found")
    
    # Test stopwords
    nltk.data.find('corpora/stopwords')
    print("[OK] stopwords corpus found")
    
    # Quick functionality test
    from nltk.tokenize import sent_tokenize
    test_text = "This is a test. It has two sentences."
    sentences = sent_tokenize(test_text)
    print(f"[OK] Sentence tokenization works: {len(sentences)} sentences found")
    
except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)
'@

poetry run python -c $verifyScript

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== NLTK Setup Complete ===" -ForegroundColor Green
    Write-Host "TORI now has sentence-level text processing capabilities!" -ForegroundColor Cyan
    Write-Host "`nYou can now use NLTK features in the ingest pipeline:" -ForegroundColor Gray
    Write-Host "  - Sentence tokenization with sent_tokenize()" -ForegroundColor Gray
    Write-Host "  - Word tokenization with word_tokenize()" -ForegroundColor Gray
    Write-Host "  - Stopword filtering with stopwords.words('english')" -ForegroundColor Gray
} else {
    Write-Host "`n=== NLTK Setup Failed ===" -ForegroundColor Red
    Write-Host "Please check the errors above and try again." -ForegroundColor Yellow
}
