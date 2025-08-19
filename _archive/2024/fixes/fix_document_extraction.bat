@echo off
echo ========================================
echo FIX: Document Extraction Dependencies
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing YAKE for keyword extraction...
pip install "yake>=0.4"

echo.
echo Installing KeyBERT for BERT-based extraction...
pip install keybert

echo.
echo Installing sentence-transformers for embeddings...
pip install sentence-transformers

echo.
echo Installing spaCy for NLP processing...
pip install "spacy>=3.7"

echo.
echo Downloading spaCy English language model...
python -m spacy download en_core_web_lg

echo.
echo ========================================
echo Testing imports...
echo ========================================
python -c "import yake; print('✅ YAKE OK')"
python -c "import keybert; print('✅ KeyBERT OK')"
python -c "import sentence_transformers; print('✅ sentence-transformers OK')"
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('✅ spaCy with en_core_web_lg OK')"

echo.
echo ✅ Document extraction dependencies installed!
echo.
echo Next: Run TORI again with:
echo python enhanced_launcher.py --no-browser
echo.
pause
