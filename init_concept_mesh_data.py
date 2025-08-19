#!/usr/bin/env python3
"""
Initialize TORI canonical concept database
Uses ONLY data/concept_db.json as the single source of truth
"""

import json
from pathlib import Path
from datetime import datetime

print("🎯 INITIALIZING TORI CANONICAL CONCEPT DATABASE")
print("=" * 60)

# Create the data directory
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

# Define canonical paths
CONCEPT_DB_PATH = data_dir / "concept_db.json"
CONCEPT_GRAPH_PATH = Path("concept_graph.json")

# Check if canonical DB already exists
if CONCEPT_DB_PATH.exists():
    print(f"\n⚠️  {CONCEPT_DB_PATH} already exists!")
    response = input("Overwrite with empty structure? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Initialization cancelled.")
        exit(0)

# Create canonical concept database structure
canonical_db = {
    "version": "2.0",
    "schema": "canonical_concept_db",
    "created_at": datetime.utcnow().isoformat(),
    "metadata": {
        "description": "Canonical concept database for TORI system",
        "total_concepts": 0,
        "last_updated": datetime.utcnow().isoformat()
    },
    "concepts": {}  # Empty but valid - will be populated by ingestion
}

# Write canonical concept database
print(f"\n📝 Creating {CONCEPT_DB_PATH}")
with open(CONCEPT_DB_PATH, 'w', encoding='utf-8') as f:
    json.dump(canonical_db, f, indent=2, ensure_ascii=False)
print("✅ Created canonical concept database")

# Create or update concept_graph.json for edges only
graph_structure = {
    "version": "2.0",
    "schema": "concept_graph_edges_only",
    "created_at": datetime.utcnow().isoformat(),
    "metadata": {
        "description": "Graph edges/relationships between concepts",
        "note": "Concept entities are stored in data/concept_db.json",
        "total_edges": 0
    },
    "edges": []  # Empty but valid
}

print(f"\n📝 Creating {CONCEPT_GRAPH_PATH}")
with open(CONCEPT_GRAPH_PATH, 'w', encoding='utf-8') as f:
    json.dump(graph_structure, f, indent=2, ensure_ascii=False)
print("✅ Created concept graph (edges only)")

# Clean up any legacy files
print("\n🧹 Checking for legacy files...")
legacy_files = [
    "concept_mesh_diffs.json",
    "concept_mesh_data.json",
    "soliton_concept_memory.json",
    "data/concept_mesh/concepts.json"
]

for legacy_file in legacy_files:
    legacy_path = Path(legacy_file)
    if legacy_path.exists():
        print(f"  ⚠️  Found legacy file: {legacy_file}")
        print("     Run migrate_concepts.py to properly archive these files")

print("\n" + "=" * 60)
print("✅ TORI canonical concept database initialized!")
print("\n📁 File Structure:")
print(f"  • {CONCEPT_DB_PATH} - The ONE source for all concepts")
print(f"  • {CONCEPT_GRAPH_PATH} - Graph edges/relationships only")
print("\n📌 Remember: ALL concept data goes in data/concept_db.json")
print("    No other files should store concept entities!")
print("\n💡 Next steps:")
print("  1. Run migrate_concepts.py if you have existing data")
print("  2. Use only data/concept_db.json in all your code")
print("  3. Store only edges in concept_graph.json")
