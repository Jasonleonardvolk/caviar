@echo off
echo Testing Concept Mesh and Lattice Without Launching TORI > "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo ============================================== >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo Test run at %date% %time% >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo. >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"

cd /d "C:\Users\jason\Desktop\tori\kha"

poetry run python -c "import sys; sys.path.insert(0, '.'); from api.concept_mesh import mesh, _concepts_view; concepts = _concepts_view(); print(f'Concept mesh has {len(concepts)} concepts'); print('Concepts:', list(concepts.keys())[:5])" >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt" 2>&1

echo. >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo Testing lattice rebuild... >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"

poetry run python -c "import sys; sys.path.insert(0, '.'); from api.lattice_routes import rebuild_lattice; import asyncio; result = asyncio.run(rebuild_lattice(full=True)); print(f'Lattice rebuild result: {result}')" >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt" 2>&1

echo. >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo Test complete. >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"

type "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
