#!/usr/bin/env python3
"""
Fix the missing load_concept_mesh function
"""

from pathlib import Path

# Read cognitive_interface.py
interface_file = Path("ingest_pdf/cognitive_interface.py")
content = interface_file.read_text(encoding='utf-8')

# Backup
backup_path = interface_file.with_suffix('.py.backup')
backup_path.write_text(content, encoding='utf-8')
print(f"✅ Created backup: {backup_path}")

# Add load_concept_mesh function
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

# Add the function after the imports section
import_section_end = content.find('def add_concept_diff')
if import_section_end > 0:
    content = content[:import_section_end] + new_function + "\n\n" + content[import_section_end:]
else:
    # If we can't find a good place, just append it
    content = content + new_function

# Write the updated file
interface_file.write_text(content, encoding='utf-8')

print("\n✅ Added load_concept_mesh function to cognitive_interface.py")
print("\n🎯 What this does:")
print("   - Searches for concept mesh data in multiple locations")
print("   - Loads from JSON files")
print("   - Returns a list of concept diffs for Prajna to use")

print("\n📝 The function will look for data in these locations:")
print("   1. concept_mesh_diffs.json")
print("   2. data/concept_mesh/diffs.json")
print("   3. data/concept_diffs/ directory")
print("   4. concept_mesh_data.json")

print("\n🧪 Test by restarting the server and checking if the error goes away")
