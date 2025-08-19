@echo off
echo ========================================
echo FIX: OCR Libraries for Scanned PDFs
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing Python OCR packages...
pip install pytesseract pdf2image pillow

echo.
echo ========================================
echo IMPORTANT: Tesseract Binary Required!
echo ========================================
echo.
echo For OCR to work, you also need to install Tesseract-OCR:
echo.
echo 1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
echo    Direct link: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe
echo.
echo 2. Run the installer and note the installation path
echo    (Default: C:\Program Files\Tesseract-OCR)
echo.
echo 3. Add Tesseract to your PATH or set in code:
echo    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
echo.
echo ========================================
echo Testing Python imports...
echo ========================================
python -c "import pytesseract; print('✅ pytesseract OK')"
python -c "import pdf2image; print('✅ pdf2image OK')"
python -c "from PIL import Image; print('✅ Pillow OK')"

echo.
echo ========================================
echo Quick Tesseract Check...
echo ========================================
where tesseract >nul 2>&1 && (
    echo ✅ Tesseract found in PATH
    tesseract --version
) || (
    echo ❌ Tesseract NOT found in PATH
    echo Please install Tesseract-OCR binary!
)

echo.
echo Next steps:
echo 1. If Tesseract not found, download and install from link above
echo 2. Add to PATH or configure in your code
echo 3. OCR will work for scanned PDFs
echo.
echo Note: OCR is optional - TORI works fine without it for regular PDFs
echo.
pause
