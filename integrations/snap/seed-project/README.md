# Snap Lens Studio — Seed Project

## What to Export from CAVIAR
- `exports/templates/concept.glb` (scene)
- `exports/textures_ktx2/*.ktx2` (materials)

## Import Steps
1) Open Lens Studio (latest).
2) File → New → Empty Project.
3) Add **3D Object** → Import `concept.glb`.
4) Add **Material** → KTX2 textures (requires KTX2 extension in Lens Studio if not built-in).
5) Wire behavior: head binding / device tracking as needed.
6) Publish to My Lenses; validate on-device.

## Notes
- Keep draw calls ≤ mobile budgets.
- Prefer power-of-two texture sizes for best compression/perf.