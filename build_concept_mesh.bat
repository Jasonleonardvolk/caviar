@echo off
echo ========================================
echo Building Concept Mesh Rust Modules
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing maturin if needed...
pip install maturin

echo.
echo [1/2] Building Penrose (similarity engine)...
cd concept_mesh\penrose_rs
maturin develop --release
cd ..\..

echo.
echo [2/2] Creating concept_mesh_rs stub...
echo Since concept_mesh_rs doesn't have PyO3 bindings yet,
echo we'll use the Python implementation.

echo.
echo Testing imports...
python -c "from concept_mesh.similarity import penrose; print('✅ Penrose available')"
python -c "from concept_mesh import load_mesh; print('✅ Concept mesh loader available')"

echo.
echo ========================================
echo ✅ Build Complete!
echo ========================================
echo.
echo The Penrose similarity engine is built.
echo Concept mesh will use Python implementation for now.
echo.
pause
