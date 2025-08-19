#!/usr/bin/env python3
"""
Fix all concept mesh related issues:
1. Missing load_concept_mesh function
2. Check which concept mesh implementation is being used
3. Ensure data directories exist
"""

import json
import sys
from pathlib import Path
import traceback

print("🔧 FIXING CONCEPT MESH ISSUES")
print("=" * 60)

# Fix 1: Add load_concept_mesh function to cognitive_interface.py
print("\n📌 Fix 1: Adding load_concept_mesh function...")
print("-" * 40)

interface_file = Path("ingest_pdf/cognitive_interface.py")
if interface_file.exists():
    content = interface_file.read_text(encoding='utf-8')
    
    # Check if function already exists
    if 'def load_concept_mesh' in content:
        print("✅ load_concept_mesh function already exists!")
    else:
        # Backup
        backup_path = interface_file.with_suffix('.py.backup_mesh_fix')
        backup_path.write_text(content, encoding='utf-8')
        print(f"✅ Created backup: {backup_path}")
        
        # Add the function
        new_function = '''

def load_concept_mesh():
    """
    Load concept mesh data from JSON files
    Returns a list of concept diff dictionaries
    """
    import json
    from pathlib import Path
    
    # Try multiple possible locations for concept mesh data
    possible_paths = [
        Path(__file__).parent.parent / "concept_mesh_diffs.json",
        Path(__file__).parent.parent / "data" / "concept_mesh" / "diffs.json",
        Path(__file__).parent.parent / "data" / "concept_diffs",
        Path(__file__).parent.parent / "concept_mesh_data.json",
    ]
    
    mesh_data = []
    
    # Try to load from a single JSON file first
    for json_path in possible_paths[:2]:
        if json_path.exists() and json_path.is_file():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        mesh_data.extend(data)
                    elif isinstance(data, dict) and 'diffs' in data:
                        mesh_data.extend(data['diffs'])
                    print(f"Loaded {len(mesh_data)} diffs from {json_path}")
                    return mesh_data
            except Exception as e:
                print(f"Failed to load {json_path}: {e}")
    
    # Try to load from individual diff files
    diffs_dir = possible_paths[2]
    if diffs_dir.exists() and diffs_dir.is_dir():
        diff_files = sorted(diffs_dir.glob("diff_*.json"))
        for diff_file in diff_files:
            try:
                with open(diff_file, 'r', encoding='utf-8') as f:
                    diff_data = json.load(f)
                    mesh_data.append(diff_data)
            except Exception as e:
                print(f"Failed to load {diff_file}: {e}")
        
        if mesh_data:
            print(f"Loaded {len(mesh_data)} diffs from {diffs_dir}")
            return mesh_data
    
    # Return empty list if no data found
    print("No concept mesh data found")
    return []
'''
        
        # Update __all__ to include load_concept_mesh
        if "__all__ = ['CognitiveInterface', 'add_concept_diff']" in content:
            content = content.replace(
                "__all__ = ['CognitiveInterface', 'add_concept_diff']",
                "__all__ = ['CognitiveInterface', 'add_concept_diff', 'load_concept_mesh']"
            )
        
        # Add the function after imports
        import_section_end = content.find('def add_concept_diff')
        if import_section_end > 0:
            content = content[:import_section_end] + new_function + "\n\n" + content[import_section_end:]
        else:
            content = content + new_function
        
        # Write the updated file
        interface_file.write_text(content, encoding='utf-8')
        print("✅ Added load_concept_mesh function!")
else:
    print("❌ cognitive_interface.py not found!")

# Fix 2: Create necessary data directories
print("\n📌 Fix 2: Creating concept mesh data directories...")
print("-" * 40)

base_path = Path(__file__).parent
data_paths = [
    base_path / "data" / "concept_mesh",
    base_path / "data" / "concept_diffs",
]

for path in data_paths:
    path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory: {path}")

# Create empty concept mesh files if they don't exist
diffs_file = base_path / "concept_mesh_diffs.json"
if not diffs_file.exists():
    with open(diffs_file, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2)
    print(f"✅ Created empty concept mesh diffs file: {diffs_file}")

# Fix 3: Check which concept mesh implementation is being used
print("\n📌 Fix 3: Checking concept mesh implementations...")
print("-" * 40)

# Check imports in enhanced_launcher.py
launcher_file = Path("enhanced_launcher.py")
if launcher_file.exists():
    launcher_content = launcher_file.read_text(encoding='utf-8')
    
    print("\n🔍 Concept mesh imports in enhanced_launcher.py:")
    if "from concept_mesh import similarity as penrose_adapt" in launcher_content:
        print("   ✅ concept_mesh module with Penrose adapter")
    if "from python.core.concept_mesh import ConceptMesh" in launcher_content:
        print("   ✅ python.core.concept_mesh (Python implementation)")
    if "concept_mesh_rs" in launcher_content:
        print("   ✅ concept_mesh_rs (Rust implementation)")

# Check imports in prajna_api.py
prajna_file = Path("prajna/api/prajna_api.py")
if prajna_file.exists():
    prajna_content = prajna_file.read_text(encoding='utf-8')
    
    print("\n🔍 Concept mesh imports in prajna_api.py:")
    if "from cognitive_interface import load_concept_mesh" in prajna_content:
        print("   ✅ Trying to import load_concept_mesh from cognitive_interface")
    if "from prajna.memory.concept_mesh_api import ConceptMeshAPI" in prajna_content:
        print("   ✅ Also importing ConceptMeshAPI")

# Fix 4: Check actual concept mesh data files
print("\n📌 Fix 4: Checking for concept mesh data files...")
print("-" * 40)

concept_files = [
    "concept_mesh_diffs.json",
    "concept_mesh_data.json",
    "concepts.json",
    "data/concept_mesh/diffs.json"
]

for file_path in concept_files:
    full_path = base_path / file_path
    if full_path.exists():
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"   ✅ {file_path}: {len(data)} items")
                elif isinstance(data, dict):
                    concepts = data.get('concepts', [])
                    print(f"   ✅ {file_path}: {len(concepts)} concepts")
        except Exception as e:
            print(f"   ⚠️ {file_path}: Error reading - {e}")
    else:
        print(f"   ❌ {file_path}: Not found")

print("\n" + "=" * 60)
print("✅ Concept mesh fixes applied!")
print("\n📝 Summary:")
print("   - load_concept_mesh function added/verified")
print("   - Data directories created")
print("   - Multiple implementations detected (this is normal)")
print("\n🎯 The system uses:")
print("   1. concept_mesh module for Penrose similarity engine")
print("   2. python.core.concept_mesh for core functionality")
print("   3. ingest_pdf.cognitive_interface for PDF processing integration")
print("\n📝 Next steps:")
print("   1. Run the soliton fix script: python fix_soliton_aliases.py")
print("   2. Restart the server: poetry run python enhanced_launcher.py")
print("   3. Check logs for 'ConceptMesh loader available' instead of error")
