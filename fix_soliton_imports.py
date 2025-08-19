#!/usr/bin/env python3
"""
Simple fix for soliton_memory imports
"""

import os
import re

print("🔧 Fixing soliton_memory.py imports...\n")

# Fix the import in soliton_memory.py
file_path = "mcp_metacognitive/core/soliton_memory.py"

if os.path.exists(file_path):
    print(f"Found {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check current state
    if "CONCEPT_MESH_AVAILABLE = False" in content:
        print("  ℹ️  Concept mesh is already disabled in this file")
    else:
        # Make a backup
        backup_path = file_path + ".backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Created backup: {backup_path}")
        
        # Fix the imports to ensure it uses the stub
        original = content
        
        # Make sure it's not trying to import from non-existent interface module
        content = re.sub(
            r'from concept_mesh_rs\.interface import ConceptMesh',
            'from concept_mesh_rs import ConceptMesh',
            content
        )
        
        # Same for types
        content = re.sub(
            r'from concept_mesh_rs\.types import (.+)',
            r'from concept_mesh_rs import \1',
            content
        )
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✅ Fixed imports")
        else:
            print("  ℹ️  No changes needed")
else:
    print(f"❌ File not found: {file_path}")

# Also check if there's a similar issue in ingest_pdf
ingest_file = "ingest_pdf/pipeline/quality.py"
if os.path.exists(ingest_file):
    print(f"\nChecking {ingest_file}...")
    
    with open(ingest_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "concept_mesh" in content.lower():
        print("  ℹ️  Contains concept_mesh references")
        # You can add similar fixes here if needed

print("\n✅ Import fixes complete!")
print("\nTesting import...")

try:
    # Add the current directory to Python path
    import sys
    sys.path.insert(0, os.getcwd())
    
    from concept_mesh_rs import ConceptMesh, ConceptMeshLoader
    print("✅ concept_mesh_rs imports successfully!")
    
    # Test instantiation
    mesh = ConceptMesh()
    loader = ConceptMeshLoader()
    print("✅ Objects can be instantiated!")
    
except Exception as e:
    print(f"❌ Import test failed: {e}")

print("\nYou can now run: poetry run python enhanced_launcher.py")
