@echo off
echo Testing ConceptMesh Singleton Pattern > "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo ============================================== >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo Test run at %date% %time% >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo. >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"

cd /d "C:\Users\jason\Desktop\tori\kha"

echo Testing singleton behavior... >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
poetry run python -c "import sys; sys.path.insert(0, '.'); from python.core.concept_mesh import ConceptMesh; m1 = ConceptMesh.instance(); m2 = ConceptMesh.instance(); print(f'Same instance: {m1 is m2}'); print(f'Instance 1 concepts: {len(m1.concepts)}'); print(f'Instance 2 concepts: {len(m2.concepts)}')" >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt" 2>&1

echo. >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo Testing API concept loading... >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
poetry run python -c "import sys; sys.path.insert(0, '.'); from api.concept_mesh import mesh, _concepts_view; concepts = _concepts_view(); print(f'API mesh has {len(concepts)} concepts'); if concepts: first_id = list(concepts.keys())[0]; print(f'First concept ID: {first_id}'); print(f'Checking ID stability...')" >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt" 2>&1

echo. >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo Testing lattice rebuild with singleton... >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
poetry run python -c "import sys; sys.path.insert(0, '.'); from api.lattice_routes import rebuild_lattice; import asyncio; result = asyncio.run(rebuild_lattice(full=True)); print(f'Lattice rebuild: oscillators_created={result[\"oscillators_created\"]}')" >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt" 2>&1

echo. >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
echo Test complete. >> "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"

type "C:\Users\jason\Desktop\tori\kha\goawaybitch.txt"
