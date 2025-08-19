#!/usr/bin/env python3
"""
Update all concept mesh references to use the canonical path: concept_mesh/data.json
"""

import os
import re
from pathlib import Path

# Define the canonical path
CANONICAL_PATH = "concept_mesh/data.json"

# Old paths that need to be replaced
OLD_PATHS = [
    "concept_mesh_data.json",
    "concepts.json",
    "concept_mesh/concepts.json",
    "data/concept_mesh/concepts.json",
    "concept_mesh_diffs.json",
]

# Files to update
FILES_TO_UPDATE = [
    "enhanced_launcher.py",
    "ingest_pdf/cognitive_interface.py",
    "concept_mesh/loader.py",
    "init_concept_mesh_data.py",
    "init_concept_mesh_data_simple.py",
    "prajna/api/prajna_api.py",
]

def update_file_references(file_path: Path):
    """Update concept mesh references in a single file"""
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        updated = False
        
        # Replace direct string references
        for old_path in OLD_PATHS:
            if old_path in content:
                # Be careful not to replace if it's already the canonical path
                if old_path != "concept_mesh/data.json":
                    content = content.replace(f'"{old_path}"', f'"{CANONICAL_PATH}"')
                    content = content.replace(f"'{old_path}'", f"'{CANONICAL_PATH}'")
                    updated = True
                    print(f"  ✅ Replaced references to {old_path}")
        
        # Update path constructions
        replacements = [
            (r'concept_mesh_dir\s*/\s*"concepts\.json"', 'concept_mesh_dir / "data.json"'),
            (r'concept_mesh_dir\s*/\s*\'concepts\.json\'', 'concept_mesh_dir / "data.json"'),
            (r'"concept_mesh"\s*/\s*"concepts\.json"', '"concept_mesh" / "data.json"'),
            (r"'concept_mesh'\s*/\s*'concepts\.json'", "'concept_mesh' / 'data.json'"),
        ]
        
        for pattern, replacement in replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                updated = True
                print(f"  ✅ Updated path construction: {pattern}")
        
        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {file_path}")
            return True
        else:
            print(f"  ℹ️  No changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    print("🔧 UPDATING ALL CONCEPT MESH REFERENCES TO CANONICAL PATH")
    print("=" * 60)
    print(f"📍 Canonical path: {CANONICAL_PATH}")
    print()
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Update each file
    updated_count = 0
    for file_name in FILES_TO_UPDATE:
        file_path = Path(file_name)
        print(f"\n📝 Checking {file_path}...")
        if update_file_references(file_path):
            updated_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Updated {updated_count} files")
    
    # Create the canonical file if it doesn't exist
    canonical_file = Path(CANONICAL_PATH)
    if not canonical_file.exists():
        print(f"\n📝 Creating canonical file: {canonical_file}")
        canonical_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        from datetime import datetime
        
        initial_data = {
            "concepts": [],
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "description": "Canonical concept mesh data storage",
                "format": "This is the authoritative concept storage for TORI"
            }
        }
        
        with open(canonical_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2)
        
        print("✅ Created canonical concept mesh file")
    else:
        print(f"\n✅ Canonical file already exists: {canonical_file}")
    
    print("\n⚠️  IMPORTANT: Remember to:")
    print("  1. Delete old concept files (concept_mesh_data.json, concepts.json)")
    print("  2. Update any other files that reference concept storage")
    print("  3. Restart TORI to use the new canonical path")

if __name__ == "__main__":
    main()
