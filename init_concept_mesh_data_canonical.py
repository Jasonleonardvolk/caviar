#!/usr/bin/env python3
"""
Initialize concept mesh with proper data structure
Using the CANONICAL path: concept_mesh/data.json
"""

import json
from pathlib import Path
from datetime import datetime

print("🔧 INITIALIZING CANONICAL CONCEPT MESH DATA")
print("=" * 60)

# Create the concept_mesh directory
concept_mesh_dir = Path("concept_mesh")
concept_mesh_dir.mkdir(parents=True, exist_ok=True)

# Initialize data.json with proper structure (CANONICAL FILE)
data_file = concept_mesh_dir / "data.json"

# Create initial empty but valid structure
initial_data = {
    "concepts": [],  # Array of concept objects
    "metadata": {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "description": "Canonical concept mesh data storage",
        "format": "This is the authoritative concept storage for TORI"
    }
}

print(f"\n📝 Creating CANONICAL file: {data_file}")
with open(data_file, 'w', encoding='utf-8') as f:
    json.dump(initial_data, f, indent=2)

print("✅ Created concept_mesh/data.json with valid structure")

# Clean up old files if they exist
old_files = [
    Path("concept_mesh_data.json"),
    Path("concepts.json"),
    Path("concept_mesh_diffs.json"),
    Path("data/concept_mesh/concepts.json"),
]

print("\n🧹 Cleaning up old concept files...")
for old_file in old_files:
    if old_file.exists():
        print(f"  ❌ Found deprecated file: {old_file} (should be removed)")
    else:
        print(f"  ✅ {old_file} not found (good)")

print("\n" + "=" * 60)
print("✅ Canonical concept mesh data initialized!")
print(f"\n📍 CANONICAL PATH: {data_file.absolute()}")
print("\n⚠️  All subsystems should reference ONLY this file!")
print("⚠️  Remove any references to old concept files in your code")
print("\n" + "=" * 60)
