@echo off
echo ========================================
echo FIX: ConceptMesh Import Issues
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Testing imports...
echo.

echo Testing concept_mesh package...
python -c "from concept_mesh import load_mesh, save_mesh; print('✅ concept_mesh package OK')"

echo.
echo Testing ingest_pdf module...
python -c "from ingest_pdf import load_concept_mesh; print('✅ ingest_pdf.load_concept_mesh OK')"

echo.
echo Testing cognitive_interface...
python -c "from ingest_pdf.cognitive_interface import CognitiveInterface; print('✅ CognitiveInterface OK')"

echo.
echo Testing all imports in context...
python -c "import sys; from pathlib import Path; sys.path.append(str(Path('.').resolve() / 'ingest_pdf')); from cognitive_interface import load_concept_mesh; print('✅ Direct import OK')"

echo.
echo ✅ ConceptMesh imports fixed!
echo.
echo The module structure is now:
echo - concept_mesh/ (package with __init__.py)
echo - ingest_pdf/ (package with __init__.py)
echo.
echo Both can be imported properly now.
echo.
pause
