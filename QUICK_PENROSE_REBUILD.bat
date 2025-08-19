@echo off
echo Quick Penrose Rebuild and Test
echo ==============================
echo.

REM Uninstall any existing version
echo Uninstalling old version...
.venv\Scripts\python -m pip uninstall -y penrose_engine_rs

REM Change to penrose_rs directory
cd concept_mesh\penrose_rs

REM Build with explicit interpreter
echo.
echo Building with explicit venv interpreter...
maturin develop --release -i ..\..\venv\Scripts\python.exe

REM Test with same interpreter
echo.
echo Testing import...
..\..\venv\Scripts\python -c "import penrose_engine_rs, sys; print('SUCCESS: Rust backend ready from', sys.executable)"

cd ..\..
echo.
pause
