@echo off
echo.
echo TESTING CONCEPT_MESH IMPORT
echo ============================================================

cd C:\Users\jason\Desktop\tori\kha

python -c "import concept_mesh_rs; print(f'Module: {concept_mesh_rs.__file__}'); print(f'ConceptMesh: {concept_mesh_rs.ConceptMesh}'); mesh = concept_mesh_rs.ConceptMesh('http://localhost:8003'); print('SUCCESS: ConceptMesh instantiation works!')"

echo.
echo Testing soliton_memory import...
python -c "import sys; sys.path.insert(0, r'mcp_metacognitive\core'); import soliton_memory; print('SUCCESS: soliton_memory imported without errors!')"

echo.
echo Ready to launch TORI!
