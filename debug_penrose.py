#!/usr/bin/env python3

import sys
from pathlib import Path

# Add concept_mesh to path
concept_mesh_path = Path("concept_mesh")
if concept_mesh_path.exists():
    sys.path.insert(0, str(concept_mesh_path))

print("Python path:", sys.path[:3])
print("Testing imports...")

print("\n1. Testing direct penrose_engine_rs import:")
try:
    import penrose_engine_rs
    print("✅ penrose_engine_rs imported successfully!")
    print(f"Functions available: {[x for x in dir(penrose_engine_rs) if not x.startswith('_')]}")
except ImportError as e:
    print(f"❌ Failed: {e}")

print("\n2. Testing concept_mesh.similarity import:")
try:
    from concept_mesh import similarity
    print("✅ concept_mesh.similarity imported successfully!")
    print(f"Backend: {similarity.BACKEND}")
    print(f"Penrose module: {similarity.penrose}")
except ImportError as e:
    print(f"❌ Failed: {e}")

print("\n3. Testing similarity module directly:")
try:
    import similarity
    print("✅ similarity imported successfully!")
    print(f"Backend: {similarity.BACKEND}")
except ImportError as e:
    print(f"❌ Failed: {e}")
