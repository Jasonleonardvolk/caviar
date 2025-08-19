"""
CRITICAL FIX: Ensure concept database is loaded correctly
"""
import os
import shutil
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("🔧 CRITICAL CONCEPT DATABASE FIX")
print("=" * 50)

# 1. ENSURE THE EXACT PATH EXISTS
target_dir = Path(r"{PROJECT_ROOT}\ingest_pdf\data")
target_dir.mkdir(parents=True, exist_ok=True)
print(f"✅ Created directory: {target_dir}")

# 2. CREATE/COPY FILES WITH EXACT NAMES
concept_file = target_dir / "concept_file_storage.json"
seed_file = target_dir / "concept_seed_universal.json"

# Look for existing concept data in various locations
possible_sources = [
    # Current directory files with wrong names
    (Path(r"{PROJECT_ROOT}\ingest_pdf\data\concept_db.json"), concept_file),
    (Path(r"{PROJECT_ROOT}\ingest_pdf\data\universal_seed.json"), seed_file),
    # Root directory
    (Path(r"{PROJECT_ROOT}\concept_mesh_data.json"), concept_file),
    (Path(r"{PROJECT_ROOT}\concepts.json"), concept_file),
    # concept-mesh directory (with hyphen)
    (Path(r"{PROJECT_ROOT}\concept-mesh\concept_mesh_data.json"), concept_file),
    (Path(r"{PROJECT_ROOT}\concept-mesh\concepts.json"), concept_file),
]

# Try to find and copy existing data
found_concepts = False
for source, target in possible_sources:
    if source.exists() and not target.exists():
        try:
            with open(source, 'r') as f:
                data = json.load(f)
                # Check if it has actual data
                if data and ((isinstance(data, list) and len(data) > 0) or 
                           (isinstance(data, dict) and len(data.get('concepts', [])) > 0)):
                    shutil.copy2(source, target)
                    print(f"✅ Copied {source.name} → {target.name}")
                    if target == concept_file:
                        found_concepts = True
                        print(f"   Found {len(data) if isinstance(data, list) else len(data.get('concepts', []))} concepts!")
        except Exception as e:
            print(f"⚠️  Error reading {source}: {e}")

# If concept file doesn't exist or is empty, create with sample data
if not concept_file.exists() or not found_concepts:
    # Create a minimal concept file
    sample_concepts = []
    with open(concept_file, 'w') as f:
        json.dump(sample_concepts, f)
    print(f"⚠️  Created empty {concept_file.name} - you need to restore your concept data!")

# Create seed file if missing
if not seed_file.exists():
    with open(seed_file, 'w') as f:
        json.dump([], f)
    print(f"✅ Created {seed_file.name}")

# 3. CLEAN UP EMPTY FILES IN concept_mesh DIRECTORY
concept_mesh_dir = Path(r"{PROJECT_ROOT}\concept_mesh")
if concept_mesh_dir.exists():
    for fname in ['concept_mesh_data.json', 'concepts.json']:
        fpath = concept_mesh_dir / fname
        if fpath.exists():
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    if not data or (isinstance(data, list) and len(data) == 0) or \
                       (isinstance(data, dict) and not data):
                        fpath.unlink()
                        print(f"✅ Removed empty {fname} from concept_mesh directory")
            except:
                pass

# 4. VERIFY FILES ARE IN PLACE
print("\n📁 Verification:")
for fpath in [concept_file, seed_file]:
    if fpath.exists():
        size = fpath.stat().st_size
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
                count = len(data) if isinstance(data, list) else len(data.get('concepts', []))
                print(f"✅ {fpath.name}: {count} items, {size} bytes")
            except:
                print(f"⚠️  {fpath.name}: exists but invalid JSON")
    else:
        print(f"❌ {fpath.name}: MISSING!")

print("\n" + "=" * 50)
print("IMPORTANT: If you see '0 items' for concept_file_storage.json,")
print("you need to find and copy your original concept database!")
print("Look for files with ~2300 concepts from your backups.")
print("=" * 50)
