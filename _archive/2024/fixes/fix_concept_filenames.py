"""
Fix TORI concept database with CORRECT filenames
"""
import json
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Step 1: Create the correct files in ingest_pdf/data/
pipeline_data_dir = Path(r"{PROJECT_ROOT}\ingest_pdf\data")
pipeline_data_dir.mkdir(parents=True, exist_ok=True)

# Create concept_file_storage.json (main database)
concept_storage_path = pipeline_data_dir / "concept_file_storage.json"
if not concept_storage_path.exists():
    with open(concept_storage_path, "w") as f:
        json.dump([], f)
    print(f"✓ Created {concept_storage_path}")
else:
    print(f"✓ Found existing {concept_storage_path}")

# Create concept_seed_universal.json (seed concepts)
seed_path = pipeline_data_dir / "concept_seed_universal.json"
if not seed_path.exists():
    with open(seed_path, "w") as f:
        json.dump([], f)
    print(f"✓ Created {seed_path}")
else:
    print(f"✓ Found existing {seed_path}")

# Step 2: Create concept_mesh directory for runtime storage
concept_mesh_dir = Path(r"{PROJECT_ROOT}\concept_mesh")
concept_mesh_dir.mkdir(exist_ok=True)

# Check if there's an old concept-mesh directory to restore from
old_mesh_dir = Path(r"{PROJECT_ROOT}\concept-mesh")
if old_mesh_dir.exists():
    print(f"\n📦 Found old concept-mesh directory at {old_mesh_dir}")
    print("You should copy any JSON files from there to the new concept_mesh directory")

print(f"""
✅ Setup complete!

The pipeline will look for:
- Main concepts: {concept_storage_path}
- Seed concepts: {seed_path}

The concept mesh will save to:
- Directory: {concept_mesh_dir}
- Files: concept_mesh_data.json and concepts.json

If you have existing concept data, copy it to these locations before running TORI.
""")
