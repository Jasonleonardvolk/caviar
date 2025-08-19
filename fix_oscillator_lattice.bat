@echo off
echo === FIXING EMPTY OSCILLATOR LATTICE ===
echo.
echo This script will:
echo 1. Enable entropy pruning
echo 2. Ingest a PDF to populate the concept mesh
echo 3. Rebuild the oscillator lattice
echo.
cd C:\Users\jason\Desktop\tori\kha
poetry run python fix_empty_lattice.py
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
echo Press any key to continue checking oscillator count...
pause
echo.
echo Checking oscillator count...
curl http://localhost:8002/api/lattice/snapshot -s | findstr oscillators
echo.
pause
