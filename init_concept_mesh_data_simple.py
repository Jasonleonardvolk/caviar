#!/usr/bin/env python3
"""
Initialize concept mesh data without Unicode
"""

import json
from pathlib import Path

print("INITIALIZING CONCEPT MESH DATA")
print("=" * 60)

# Create the data directories
data_dir = Path("data/concept_mesh")
data_dir.mkdir(parents=True, exist_ok=True)

# Initialize concepts.json with proper structure
concepts_file = data_dir / "concepts.json"

# Create initial empty but valid structure
initial_data = {
    "concepts": {},  # Empty but valid
    "version": "1.0",
    "metadata": {
        "created": "2024-01-01T00:00:00Z",
        "last_updated": "2024-01-01T00:00:00Z",
        "count": 0
    }
}

print(f"\n[*] Creating {concepts_file}")
with open(concepts_file, 'w', encoding='utf-8') as f:
    json.dump(initial_data, f, indent=2)

print("[OK] Created concepts.json with valid empty structure")

# Also create concept_mesh_diffs.json
diffs_file = Path("concept_mesh_diffs.json")
print(f"\n[*] Creating {diffs_file}")
with open(diffs_file, 'w', encoding='utf-8') as f:
    json.dump([], f, indent=2)  # Empty array is valid

print("[OK] Created concept_mesh_diffs.json")

# Create concept_mesh_data.json as well
mesh_data_file = Path("concept_mesh_data.json")
print(f"\n[*] Creating {mesh_data_file}")
with open(mesh_data_file, 'w', encoding='utf-8') as f:
    json.dump({"concepts": {}, "diffs": []}, f, indent=2)

print("[OK] Created concept_mesh_data.json")

print("\n" + "=" * 60)
print("[OK] Concept mesh data initialized!")
print("\nThe data files now have valid empty structures:")
print("  - data/concept_mesh/concepts.json")
print("  - concept_mesh_diffs.json") 
print("  - concept_mesh_data.json")
print("\nThis should prevent any '0 entries' errors.")
