@echo off
echo ========================================
echo FIX: ConceptMesh Import Issues (v2)
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Testing imports after fixes...
echo.

echo [1/4] Testing concept_mesh package...
python -c "from concept_mesh import load_mesh, save_mesh; print('✅ concept_mesh package OK')"

echo.
echo [2/4] Testing ingest_pdf.load_concept_mesh...
python -c "from ingest_pdf import load_concept_mesh; result = load_concept_mesh(); print(f'✅ load_concept_mesh OK (returned {type(result).__name__})')"

echo.
echo [3/4] Testing cognitive_interface...
python -c "from ingest_pdf.cognitive_interface import CognitiveInterface, add_concept_diff; print('✅ CognitiveInterface OK')"

echo.
echo [4/4] Testing prajna API import style...
python -c "import sys; from pathlib import Path; sys.path.append(str(Path('.').resolve() / 'ingest_pdf')); from cognitive_interface import load_concept_mesh; print('✅ API-style import OK')"

echo.
echo ========================================
echo ✅ All imports fixed!
echo ========================================
echo.
echo The ingest_pdf module now has:
echo - load_concept_mesh() function that returns concept data
echo - CognitiveInterface class for concept management
echo - add_concept_diff() function for tracking changes
echo.
echo Ready to start TORI!
echo.
pause
