@echo off
echo === SEEDING CONCEPT MESH WITH INITIAL DATA ===
echo.
echo This script will:
echo 1. Seed the concept mesh with initial data
echo 2. Rebuild the lattice to create oscillators
echo.
cd C:\Users\jason\Desktop\tori\kha
poetry run python seed_concept_mesh.py
echo.
echo === VERIFICATION STEPS ===
echo.
echo To verify the fix:
echo 1. Check the lattice snapshot:
echo    curl http://localhost:8002/api/lattice/snapshot | jq '.summary'
echo.
echo 2. Verify concept mesh count:
echo    curl http://localhost:8002/api/concept_mesh/stats
echo.
echo Press any key to exit...
pause
