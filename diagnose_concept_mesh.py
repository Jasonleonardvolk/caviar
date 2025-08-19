"""
Find and diagnose concept mesh issues
"""
import os
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("🔍 CONCEPT MESH DIAGNOSTIC")
print("=" * 50)

# Check all possible locations for concept files
locations = [
    # Pipeline expected location
    Path(r"{PROJECT_ROOT}\ingest_pdf\data\concept_file_storage.json"),
    Path(r"{PROJECT_ROOT}\ingest_pdf\data\concept_seed_universal.json"),
    
    # Concept mesh expected locations
    Path(r"{PROJECT_ROOT}\concept_mesh\concept_mesh_data.json"),
    Path(r"{PROJECT_ROOT}\concept_mesh\concepts.json"),
    
    # Original location (with hyphen)
    Path(r"{PROJECT_ROOT}\concept-mesh\concept_mesh_data.json"),
    Path(r"{PROJECT_ROOT}\concept-mesh\concepts.json"),
    
    # Root directory
    Path(r"{PROJECT_ROOT}\concept_mesh_data.json"),
    Path(r"{PROJECT_ROOT}\concepts.json"),
]

print("📁 Checking all concept file locations:\n")

found_data = False
for fpath in locations:
    if fpath.exists():
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    count = len(data.get('concepts', []))
                else:
                    count = 0
                
                size = fpath.stat().st_size
                print(f"✅ FOUND: {fpath}")
                print(f"   Items: {count}, Size: {size} bytes")
                
                if count > 1000:  # Likely the real database
                    found_data = True
                    print(f"   🎯 THIS LOOKS LIKE YOUR CONCEPT DATABASE!")
                print()
        except Exception as e:
            print(f"❌ ERROR reading {fpath}: {e}\n")
    else:
        if "concept_file_storage" in str(fpath) or "concept_mesh_data" in str(fpath):
            print(f"❌ MISSING: {fpath}\n")

if not found_data:
    print("⚠️  NO CONCEPT DATABASE WITH SIGNIFICANT DATA FOUND!")
    print("\nYou need to find your original concept database with ~2300 concepts.")
    print("Check:")
    print("  - Backup folders")
    print("  - Previous Git commits")
    print("  - Other TORI installations")
else:
    print("\n📋 NEXT STEPS:")
    print("1. Copy the file with 2300+ concepts to BOTH:")
    print("   - str(PROJECT_ROOT / "ingest_pdf\\data\\concept_file_storage.json")
    print("   - str(PROJECT_ROOT / "concept_mesh\\concept_mesh_data.json")
    print("2. Run: python enhanced_launcher.py")
