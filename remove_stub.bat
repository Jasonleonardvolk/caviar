@echo off
echo.
echo REMOVING CONFLICTING PYTHON STUB
echo ============================================================

cd C:\Users\jason\Desktop\tori\kha

if exist concept_mesh_rs (
    echo.
    echo Removing concept_mesh_rs Python directory...
    rmdir /s /q concept_mesh_rs
    echo Removed conflicting directory
) else (
    echo No conflicting directory found
)

echo.
echo Now testing the import...
python -c "import concept_mesh_rs; print(f'Module location: {concept_mesh_rs.__file__}'); mesh = concept_mesh_rs.ConceptMesh('http://localhost:8003/api/mesh'); print('SUCCESS: ConceptMesh works!')"

echo.
echo Done! Now run:
echo poetry run python enhanced_launcher.py
