"""
Copy concept data to correct filenames
"""
import shutil
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

data_dir = Path(r"{PROJECT_ROOT}\ingest_pdf\data")

# Copy concept_database.json to concept_file_storage.json
src = data_dir / "concept_database.json"
dst = data_dir / "concept_file_storage.json"

if src.exists():
    shutil.copy2(src, dst)
    print(f"✓ Copied {src.name} → {dst.name}")
else:
    print(f"✗ Source file {src} not found")

# Check if we need to copy universal_seed.json to concept_seed_universal.json
src_seed = data_dir / "universal_seed.json"
dst_seed = data_dir / "concept_seed_universal.json"

if src_seed.exists() and src_seed.stat().st_size > 10:  # If it has content
    shutil.copy2(src_seed, dst_seed)
    print(f"✓ Copied {src_seed.name} → {dst_seed.name}")

print("\n✅ Concept files are now using the correct names!")
print("\nRun: python enhanced_launcher.py")
