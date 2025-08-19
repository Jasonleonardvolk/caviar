@echo off
echo.
echo REMOVING ALL CONCEPT_MESH_RS STUBS
echo ============================================================

cd C:\Users\jason\Desktop\tori\kha

echo Removing concept_mesh\concept_mesh_rs directory...
if exist concept_mesh\concept_mesh_rs (
    rmdir /s /q concept_mesh\concept_mesh_rs
    echo Removed concept_mesh\concept_mesh_rs
)

echo.
echo Checking for any other concept_mesh_rs directories...
dir /s /b concept_mesh_rs 2>nul

echo.
echo Testing import...
python -c "import concept_mesh_rs; print(f'Module: {concept_mesh_rs.__file__}')"

echo.
echo Done!
