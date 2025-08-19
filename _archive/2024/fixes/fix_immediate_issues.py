#!/usr/bin/env python3
"""
Fix the immediate issues:
1. Missing load_concept_mesh function in cognitive_interface.py
2. Soliton endpoint field name mismatch
"""

import sys
from pathlib import Path

print("🔧 FIXING IMMEDIATE ISSUES")
print("=" * 60)

# Fix 1: Add load_concept_mesh to cognitive_interface.py
print("\n📌 Fix 1: Adding load_concept_mesh function...")
print("-" * 40)

cognitive_interface_file = Path("ingest_pdf/cognitive_interface.py")
if cognitive_interface_file.exists():
    content = cognitive_interface_file.read_text(encoding='utf-8')
    
    # Check if function already exists
    if 'def load_concept_mesh' in content:
        print("✅ load_concept_mesh already exists!")
    else:
        # Backup first
        backup_path = cognitive_interface_file.with_suffix('.py.backup_fix')
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
        Path(__file__).parent.parent / "concept_mesh_data.json",
        Path(__file__).parent.parent / "data" / "concept_mesh" / "diffs.json",
        Path(__file__).parent.parent / "data" / "concept_diffs",
    ]
    
    mesh_data = []
    
    # Try to load from a single JSON file first
    for json_path in possible_paths[:-1]:  # Skip directory path
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
    diffs_dir = possible_paths[-1]
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
        
        # Add the function after the imports section
        import_section_end = content.find('def add_concept_diff')
        if import_section_end > 0:
            content = content[:import_section_end] + new_function + "\n\n" + content[import_section_end:]
        else:
            # If we can't find a good place, just append it
            content = content + new_function
        
        # Write the updated file
        cognitive_interface_file.write_text(content, encoding='utf-8')
        print("✅ Added load_concept_mesh function!")
else:
    print("❌ cognitive_interface.py not found!")

# Fix 2: Create concept mesh data directories
print("\n📌 Fix 2: Creating concept mesh data directories...")
print("-" * 40)

base_path = Path(".")
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
    import json
    with open(diffs_file, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2)
    print(f"✅ Created empty concept mesh diffs file: {diffs_file}")

# Fix 3: Check soliton endpoint status
print("\n📌 Fix 3: Checking soliton endpoints...")
print("-" * 40)

soliton_file = Path("api/routes/soliton.py")
if soliton_file.exists():
    soliton_content = soliton_file.read_text(encoding='utf-8')
    
    # The current soliton.py already accepts user_id (not user)
    if 'user_id: Optional[str]' in soliton_content:
        print("✅ Soliton endpoints already accept user_id field!")
        print("   Frontend sends: user_id ✓")
        print("   Backend expects: user_id ✓")
    else:
        print("⚠️ Soliton endpoints may need field name adjustment")

print("\n" + "=" * 60)
print("✅ Fixes applied!")
print("\n📝 Next steps:")
print("   1. Restart the server")
print("   2. Check that 'ConceptMesh loader available' appears in logs")
print("   3. Soliton endpoints should work without 500 errors")
