@echo off
REM Quick Setup for Cognitive Interface and Concept Mesh
REM =====================================================

echo ========================================
echo Setting up Cognitive Interface and Concept Mesh
echo ========================================
echo.

REM Step 1: Set Project Root
cd /d C:\Users\jason\Desktop\tori\kha
set PROJECT_ROOT=%cd%
echo Project root: %PROJECT_ROOT%

REM Step 2: Set PYTHONPATH
set PYTHONPATH=%PROJECT_ROOT%
echo PYTHONPATH set to: %PYTHONPATH%
echo.

REM Step 3: Create __init__.py if needed
if not exist "ingest_pdf\__init__.py" (
    echo Creating __init__.py in ingest_pdf...
    echo. > ingest_pdf\__init__.py
)

REM Step 4: Test imports
echo Testing imports...
python -c "import sys; sys.path.insert(0, r'%PROJECT_ROOT%'); from ingest_pdf.cognitive_interface import add_concept_diff; print('SUCCESS: cognitive_interface imported!')" 2>nul
if %errorlevel% neq 0 (
    echo FAILED to import cognitive_interface
    echo Trying alternative import...
    python -c "import sys; sys.path.insert(0, r'%PROJECT_ROOT%\ingest_pdf'); import cognitive_interface; print('SUCCESS: alternative import worked!')" 2>nul
)
echo.

REM Step 5: Install concept-mesh if needed
echo Checking for concept-mesh...
python -c "import concept_mesh" 2>nul
if %errorlevel% neq 0 (
    echo concept_mesh not found, installing...
    pip install concept-mesh-client 2>nul
    if %errorlevel% neq 0 (
        echo concept-mesh-client not available on PyPI
        echo Creating mock concept_mesh...
        mkdir concept_mesh 2>nul
        echo # Mock ConceptMeshConnector > concept_mesh\__init__.py
        echo class ConceptMeshConnector: >> concept_mesh\__init__.py
        echo     def __init__(self, url=None): self.url = url >> concept_mesh\__init__.py
        echo     def connect(self): return True >> concept_mesh\__init__.py
    )
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the Cognitive Interface service:
echo   python -m uvicorn ingest_pdf.cognitive_interface:app --port 5173
echo.
echo Or use the start script:
echo   start_cognitive_interface.bat
echo.
echo To test in Python:
echo   python
echo   ^>^>^> from ingest_pdf.cognitive_interface import add_concept_diff
echo   ^>^>^> from concept_mesh import ConceptMeshConnector
echo.
pause
