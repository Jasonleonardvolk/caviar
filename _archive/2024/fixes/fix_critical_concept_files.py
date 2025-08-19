#!/usr/bin/env python3
"""
Quick Fix for Critical Concept File References
Updates the most important files to use canonical sources
"""

import re
from pathlib import Path

# Define the critical files that MUST be fixed
CRITICAL_FILES = [
    "prajna/api/prajna_api.py",
    "prajna_api.py",  # If different from above
    "prajna/memory/concept_mesh_api.py",
    "prajna/memory/soliton_interface.py",
    "ingest_pdf/cognitive_interface.py",
    "conversations/cognitive_interface.py"
]

# Replacement patterns
REPLACEMENTS = [
    # Replace soliton memory references
    (r'soliton_concept_memory\.json', 'data/concept_db.json'),
    (r'"soliton_concept_memory\.json"', '"data/concept_db.json"'),
    (r"'soliton_concept_memory\.json'", "'data/concept_db.json'"),
    
    # Replace concept mesh data references
    (r'concept_mesh_data\.json', 'data/concept_db.json'),
    (r'"concept_mesh_data\.json"', '"data/concept_db.json"'),
    (r"'concept_mesh_data\.json'", "'data/concept_db.json'"),
    
    # Replace concept mesh diffs references (comment out or remove)
    (r'concept_mesh_diffs\.json', 'data/concept_db.json'),
    
    # Replace nested concept mesh path
    (r'data/concept_mesh/concepts\.json', 'data/concept_db.json'),
    
    # Update path constructions
    (r'Path\("soliton_concept_memory\.json"\)', 'Path("data/concept_db.json")'),
    (r'Path\("concept_mesh_data\.json"\)', 'Path("data/concept_db.json")'),
    
    # Update any .parent.parent patterns that reference these files
    (r'/ "soliton_concept_memory\.json"', '/ "data" / "concept_db.json"'),
    (r'/ "concept_mesh_data\.json"', '/ "data" / "concept_db.json"'),
]

def fix_file(filepath):
    """Fix a single file with all replacements"""
    path = Path(filepath)
    
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"\n🔧 Processing: {filepath}")
    
    # Read the file
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        original_content = content
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False
    
    # Apply replacements
    changes_made = 0
    for pattern, replacement in REPLACEMENTS:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            count = len(re.findall(pattern, content))
            print(f"   ✓ Replaced {count} instances of '{pattern}'")
            changes_made += count
            content = new_content
    
    # If changes were made, write the file
    if content != original_content:
        # Create backup
        backup_path = path.with_suffix(path.suffix + '.backup_canonical')
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            print(f"   📁 Backup saved to: {backup_path}")
        except Exception as e:
            print(f"   ⚠️  Could not create backup: {e}")
        
        # Write updated content
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   ✅ File updated successfully ({changes_made} changes)")
            return True
        except Exception as e:
            print(f"   ❌ Error writing file: {e}")
            return False
    else:
        print(f"   ℹ️  No changes needed")
        return True

def main():
    print("🚀 CRITICAL FILE FIX TOOL")
    print("="*60)
    print("This will update the most critical files to use canonical sources.")
    print("\nFiles to be updated:")
    for file in CRITICAL_FILES:
        print(f"  • {file}")
    
    print("\nReplacements:")
    print("  • soliton_concept_memory.json → data/concept_db.json")
    print("  • concept_mesh_data.json → data/concept_db.json")
    print("  • concept_mesh_diffs.json → data/concept_db.json")
    print("  • data/concept_mesh/concepts.json → data/concept_db.json")
    
    response = input("\nProceed with fixes? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Process each file
    success_count = 0
    for filepath in CRITICAL_FILES:
        if fix_file(filepath):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"✅ Successfully updated {success_count}/{len(CRITICAL_FILES)} files")
    
    if success_count == len(CRITICAL_FILES):
        print("\n🎉 All critical files updated!")
        print("\nNext steps:")
        print("1. Test the system to ensure everything works")
        print("2. Run audit_concept_usage.py again to check progress")
        print("3. Fix remaining non-critical files")
    else:
        print("\n⚠️  Some files could not be updated.")
        print("Please check the errors above and fix manually.")

if __name__ == "__main__":
    main()
