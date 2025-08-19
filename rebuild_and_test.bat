@echo off
echo.
echo REBUILDING CONCEPT_MESH RUST EXTENSION
echo ============================================================

cd C:\Users\jason\Desktop\tori\kha\concept_mesh

echo Building with maturin...
maturin develop --release

echo.
echo Testing import...
cd ..
python -c "import concept_mesh_rs; print(f'SUCCESS! Module: {concept_mesh_rs.__file__}'); mesh = concept_mesh_rs.ConceptMesh('http://localhost:8003/api/mesh'); print('ConceptMesh instantiation works!')"

echo.
echo Ready to launch TORI!
