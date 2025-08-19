@echo off
echo ========================================
echo FINAL SANITY CHECK & FIXES
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo [1/6] Checking YAKE and document extraction...
python -c "import yake; print('✅ YAKE OK')" || echo ❌ YAKE missing
python -c "import keybert; print('✅ KeyBERT OK')" || echo ❌ KeyBERT missing
python -c "import sentence_transformers; print('✅ sentence-transformers OK')" || echo ❌ sentence-transformers missing
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('✅ spaCy with model OK')" || echo ❌ spaCy or model missing

echo.
echo [2/6] Checking entropy pruning...
python -c "import entropy_prune; print('✅ entropy-prune OK')" || (
    echo ⚠️ entropy-prune not installed - setting disable flag
    set TORI_DISABLE_ENTROPY_PRUNE=1
)

echo.
echo [3/6] Checking OCR (optional)...
python -c "import pytesseract; print('✅ OCR libraries OK')" || echo 🛈 OCR disabled - skipping scanned pages

echo.
echo [4/6] Checking MCP types...
python -c "from mcp.types import Tool, TextContent; print('✅ MCP types OK')" || echo ❌ MCP not installed

echo.
echo [5/6] Testing ingest_pdf imports...
python -c "from ingest_pdf import load_concept_mesh; print('✅ ingest_pdf imports OK')" || echo ❌ ingest_pdf import failed

echo.
echo [6/6] Testing concept_mesh imports...
python -c "from concept_mesh import load_mesh; print('✅ concept_mesh imports OK')" || echo ❌ concept_mesh import failed

echo.
echo ========================================
echo APPLYING FINAL FIXES...
echo ========================================
echo.

REM Install any missing critical dependencies
pip install yake keybert sentence-transformers spacy mcp --quiet

echo.
echo Ensuring spaCy model is downloaded...
python -m spacy download en_core_web_lg --quiet || echo Model already downloaded

echo.
echo ========================================
echo READY TO START TORI!
echo ========================================
echo.
echo Run: python enhanced_launcher.py --no-browser
echo.
echo Expected:
echo - ✅ Universal extraction models ready
echo - ✅ No entropy_prune error 
echo - 🛈 OCR disabled (or enabled if installed)
echo - ✅ MCP server available
echo - ✅ API health check returns 200
echo.
pause
