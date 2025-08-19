"""
Fix concept database file names to match what the pipeline expects
"""
import os
import shutil
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The pipeline expects these exact filenames
EXPECTED_CONCEPT_FILE = "concept_file_storage.json"
EXPECTED_SEED_FILE = "concept_seed_universal.json"

# Directory where pipeline looks
target_dir = Path(r"{PROJECT_ROOT}\ingest_pdf\data")
target_dir.mkdir(parents=True, exist_ok=True)

# Check for existing files with wrong names
old_concept_db = target_dir / "concept_db.json"
old_universal_seed = target_dir / "universal_seed.json"

# Copy/rename to correct names
if old_concept_db.exists():
    shutil.copy2(old_concept_db, target_dir / EXPECTED_CONCEPT_FILE)
    print(f"✓ Copied concept_db.json → {EXPECTED_CONCEPT_FILE}")
else:
    # Create empty file with correct name
    with open(target_dir / EXPECTED_CONCEPT_FILE, "w") as f:
        json.dump([], f)
    print(f"✓ Created empty {EXPECTED_CONCEPT_FILE}")

if old_universal_seed.exists():
    shutil.copy2(old_universal_seed, target_dir / EXPECTED_SEED_FILE)
    print(f"✓ Copied universal_seed.json → {EXPECTED_SEED_FILE}")
else:
    # Create empty file with correct name
    with open(target_dir / EXPECTED_SEED_FILE, "w") as f:
        json.dump([], f)
    print(f"✓ Created empty {EXPECTED_SEED_FILE}")

# Also check for concept_mesh directory with the expected files
concept_mesh_dir = Path(r"{PROJECT_ROOT}\concept_mesh")
concept_mesh_dir.mkdir(exist_ok=True)

# The ConceptMesh also looks for these files
mesh_data_file = concept_mesh_dir / "concept_mesh_data.json"
concepts_file = concept_mesh_dir / "concepts.json"

# Check if we have existing concept data to restore
existing_files = [
    Path(r"{PROJECT_ROOT}\concept_mesh_data.json"),
    Path(r"{PROJECT_ROOT}\concepts.json"),
]

for src_file in existing_files:
    if src_file.exists():
        dst_file = concept_mesh_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"✓ Restored {src_file.name} to concept_mesh directory")

# Create empty files if they don't exist
if not mesh_data_file.exists():
    with open(mesh_data_file, "w") as f:
        json.dump({}, f)
    print(f"✓ Created empty concept_mesh_data.json")

if not concepts_file.exists():
    with open(concepts_file, "w") as f:
        json.dump([], f)
    print(f"✓ Created empty concepts.json")

print(f"""
✅ Concept files fixed!

Pipeline will look for:
- {target_dir / EXPECTED_CONCEPT_FILE}
- {target_dir / EXPECTED_SEED_FILE}

ConceptMesh will look for:
- {mesh_data_file}
- {concepts_file}

Next steps:
1. If you have the original concept-mesh folder with 2304 concepts, copy its contents to:
   {concept_mesh_dir}
   
2. Run: python enhanced_launcher.py

You should see:
✅ Main concept file_storage loaded: X concepts
🌍 UNIVERSAL DATABASE READY: X total concepts
""")
