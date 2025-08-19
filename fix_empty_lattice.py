#!/usr/bin/env python3
"""
Fix Empty Oscillator Lattice Issue
==================================

This script addresses the root cause: empty concept mesh preventing oscillator creation.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

def create_initial_concepts():
    """Create some initial concepts to bootstrap the system"""
    
    # Initial concepts to populate the mesh
    initial_concepts = [
        {
            "id": "consciousness_001",
            "name": "Consciousness",
            "description": "The state of being aware of and able to think about one's existence",
            "category": "philosophy",
            "importance": 1.0,
            "phase": 0.0,
            "amplitude": 1.0
        },
        {
            "id": "intelligence_001", 
            "name": "Intelligence",
            "description": "The ability to acquire and apply knowledge and skills",
            "category": "cognitive_science",
            "importance": 0.9,
            "phase": 1.57,
            "amplitude": 0.95
        },
        {
            "id": "memory_001",
            "name": "Memory", 
            "description": "The faculty by which the mind stores and remembers information",
            "category": "neuroscience",
            "importance": 0.85,
            "phase": 3.14,
            "amplitude": 0.9
        },
        {
            "id": "perception_001",
            "name": "Perception",
            "description": "The ability to see, hear, or become aware of something through the senses",
            "category": "psychology",
            "importance": 0.8,
            "phase": 4.71,
            "amplitude": 0.85
        },
        {
            "id": "learning_001",
            "name": "Learning",
            "description": "The acquisition of knowledge or skills through experience, study, or teaching",
            "category": "education",
            "importance": 0.88,
            "phase": 2.36,
            "amplitude": 0.92
        }
    ]
    
    return initial_concepts

def fix_concept_mesh_data():
    """Fix the concept_mesh_data.json file"""
    
    concept_mesh_file = Path("concept_mesh_data.json")
    
    # Load existing data or create new
    if concept_mesh_file.exists():
        with open(concept_mesh_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    # Ensure proper structure
    if "concepts" not in data or not isinstance(data["concepts"], dict):
        data["concepts"] = {}
    
    # Add initial concepts
    initial_concepts = create_initial_concepts()
    for concept in initial_concepts:
        concept_id = concept["id"]
        data["concepts"][concept_id] = concept
        print(f"✅ Added concept: {concept['name']}")
    
    # Update metadata
    if "metadata" not in data:
        data["metadata"] = {}
    
    data["metadata"]["version"] = "1.0"
    data["metadata"]["updated_at"] = datetime.now().isoformat()
    data["metadata"]["concept_count"] = len(data["concepts"])
    
    # Save updated data
    with open(concept_mesh_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Updated concept_mesh_data.json with {len(data['concepts'])} concepts")
    return len(data["concepts"])

def fix_concepts_json():
    """Fix the concepts.json file"""
    
    concepts_file = Path("concepts.json")
    
    # Load existing data or create new
    if concepts_file.exists():
        with open(concepts_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    # Ensure proper structure
    if "concepts" not in data or not isinstance(data["concepts"], list):
        data["concepts"] = []
    
    # Add initial concepts
    initial_concepts = create_initial_concepts()
    for concept in initial_concepts:
        # Check if concept already exists
        exists = any(c.get("id") == concept["id"] for c in data["concepts"])
        if not exists:
            data["concepts"].append(concept)
    
    # Update metadata
    if "metadata" not in data:
        data["metadata"] = {}
    
    data["metadata"]["version"] = "1.0"
    data["metadata"]["updated_at"] = datetime.now().isoformat()
    data["metadata"]["concept_count"] = len(data["concepts"])
    
    # Save updated data
    with open(concepts_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated concepts.json with {len(data['concepts'])} concepts")
    return len(data["concepts"])

def enable_entropy_pruning():
    """Set environment variable to enable entropy pruning"""
    
    print("\n🔧 Enabling entropy pruning...")
    
    # Check current setting
    current = os.environ.get('TORI_DISABLE_ENTROPY_PRUNE', '1')
    if current == '1':
        print("⚠️  Entropy pruning is currently DISABLED")
        print("📝 To enable it, set: TORI_ENABLE_ENTROPY_PRUNING=1")
        print("   or unset: TORI_DISABLE_ENTROPY_PRUNE")
    else:
        print("✅ Entropy pruning is already enabled")
    
    # Create a batch file to set the environment variable
    batch_content = """@echo off
echo Setting TORI environment variables...
set TORI_ENABLE_ENTROPY_PRUNING=1
set TORI_DISABLE_ENTROPY_PRUNE=
echo ✅ Entropy pruning enabled
echo.
echo Now run: poetry run python enhanced_launcher.py
pause
"""
    
    with open("enable_entropy_pruning.bat", 'w', encoding='utf-8') as f:
        f.write(batch_content)
    
    print("📄 Created enable_entropy_pruning.bat")

def main():
    """Main fix process"""
    
    print("🔧 FIXING EMPTY OSCILLATOR LATTICE")
    print("=" * 50)
    
    # Step 1: Fix concept mesh files
    print("\n📁 Step 1: Populating concept mesh...")
    count1 = fix_concept_mesh_data()
    count2 = fix_concepts_json()
    
    # Step 2: Enable entropy pruning
    print("\n🔀 Step 2: Configuring entropy pruning...")
    enable_entropy_pruning()
    
    # Step 3: Next steps
    print("\n📋 NEXT STEPS:")
    print("=" * 50)
    print("1. Run the batch file to enable entropy pruning:")
    print("   > enable_entropy_pruning.bat")
    print()
    print("2. Restart the enhanced launcher:")
    print("   > poetry run python enhanced_launcher.py")
    print()
    print("3. Check the oscillator count (should be > 0):")
    print("   > curl http://localhost:8002/api/lattice/snapshot")
    print()
    print("4. Optional - Ingest more documents:")
    print("   > poetry run python ingest_pdfs_only_FIXED.py /path/to/pdfs")
    print()
    print("5. Optional - Force lattice rebuild:")
    print("   > curl -X POST http://localhost:8002/api/lattice/rebuild")
    
    print("\n✅ Fix script completed successfully!")
    print(f"📊 Added {count1} concepts to the mesh")
    print("🎯 The oscillator lattice should now create oscillators from these concepts")

if __name__ == "__main__":
    main()
