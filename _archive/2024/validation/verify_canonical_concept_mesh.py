#!/usr/bin/env python3
"""
Verify canonical concept mesh setup
"""

import json
from pathlib import Path
from datetime import datetime

def verify_canonical_setup():
    """Verify that the canonical concept mesh is properly set up"""
    
    print("🔍 VERIFYING CANONICAL CONCEPT MESH SETUP")
    print("=" * 60)
    
    # Check canonical file
    canonical_path = Path("concept_mesh/data.json")
    
    if canonical_path.exists():
        print(f"✅ Canonical file exists: {canonical_path.absolute()}")
        
        # Verify structure
        try:
            with open(canonical_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'concepts' in data and 'metadata' in data:
                print("✅ Canonical file has correct structure")
                print(f"   - Concepts: {len(data.get('concepts', []))} entries")
                print(f"   - Version: {data.get('metadata', {}).get('version', 'unknown')}")
            else:
                print("❌ Canonical file has incorrect structure")
                return False
                
        except Exception as e:
            print(f"❌ Error reading canonical file: {e}")
            return False
    else:
        print(f"❌ Canonical file does not exist: {canonical_path}")
        return False
    
    # Check for deprecated files
    print("\n📋 Checking for deprecated files...")
    deprecated_files = [
        Path("concept_mesh_data.json"),
        Path("concepts.json"),
        Path("concept_mesh_diffs.json"),
        Path("prajna/concept_mesh_data.json"),
    ]
    
    found_deprecated = False
    for deprecated in deprecated_files:
        if deprecated.exists():
            print(f"   ⚠️  Found deprecated file: {deprecated} (should be deleted)")
            found_deprecated = True
        else:
            print(f"   ✅ {deprecated} not found (good)")
    
    # Check code references
    print("\n📋 Checking code references...")
    files_to_check = [
        ("enhanced_launcher.py", 'concept_mesh_dir / "data.json"'),
        ("concept_mesh/loader.py", '"data.json"'),
        ("ingest_pdf/cognitive_interface.py", '"concept_mesh" / "data.json"'),
    ]
    
    all_correct = True
    for file_path, expected_ref in files_to_check:
        path = Path(file_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if expected_ref in content:
                    print(f"   ✅ {file_path}: Using canonical path")
                else:
                    print(f"   ❌ {file_path}: Not using canonical path")
                    all_correct = False
            except Exception as e:
                print(f"   ⚠️  {file_path}: Error checking - {e}")
        else:
            print(f"   ⚠️  {file_path}: File not found")
    
    print("\n" + "=" * 60)
    
    if canonical_path.exists() and all_correct and not found_deprecated:
        print("✅ CANONICAL SETUP COMPLETE!")
        print(f"\n📍 All systems are using: {canonical_path.absolute()}")
        return True
    else:
        print("⚠️  CANONICAL SETUP INCOMPLETE")
        print("\nTo complete the setup:")
        if not canonical_path.exists():
            print("  1. Run: python init_concept_mesh_data_canonical.py")
        if found_deprecated:
            print("  2. Delete all deprecated concept files listed above")
        if not all_correct:
            print("  3. Run: python update_concept_mesh_references.py")
        print("  4. Restart TORI")
        return False

if __name__ == "__main__":
    verify_canonical_setup()
