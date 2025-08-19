#!/usr/bin/env python3
"""Quick concept mesh seeder"""
import json
import sys
from pathlib import Path

def seed_concept_mesh():
    """Seed the concept mesh with initial data"""
    
    # Find concept mesh data file
    concept_mesh_file = Path("concept_mesh/data.json")
    if not concept_mesh_file.parent.exists():
        concept_mesh_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read seed data
    seed_file = Path("data/seed_concepts/quick.jsonl")
    if not seed_file.exists():
        print(f"[ERROR] Seed file not found: {seed_file}")
        return False
    
    concepts = []
    with open(seed_file, 'r') as f:
        for line in f:
            if line.strip():
                concepts.append(json.loads(line))
    
    # Create proper structure
    mesh_data = {
        "concepts": concepts,
        "metadata": {
            "version": "1.0",
            "seeded": True,
            "seed_count": len(concepts)
        }
    }
    
    # Write to concept mesh
    with open(concept_mesh_file, 'w') as f:
        json.dump(mesh_data, f, indent=2)
    
    print(f"[OK] Seeded concept mesh with {len(concepts)} concepts")
    print(f"   Written to: {concept_mesh_file}")
    return True

if __name__ == "__main__":
    success = seed_concept_mesh()
    sys.exit(0 if success else 1)
