"""
FINAL FIX - Copy the actual concept data to the right filename
"""
import shutil
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("🔧 FINAL CONCEPT DATABASE FIX")
print("=" * 50)

source = Path(r"{PROJECT_ROOT}\ingest_pdf\data\concept_database.json")
target = Path(r"{PROJECT_ROOT}\ingest_pdf\data\concept_file_storage.json")

# Backup the empty file first
if target.exists():
    shutil.move(target, target.with_suffix('.json.empty'))
    print(f"✅ Backed up empty file to {target.name}.empty")

# Copy the real data
shutil.copy2(source, target)
print(f"✅ Copied {source.name} → {target.name}")

# Verify
import json
with open(target, 'r') as f:
    data = json.load(f)
    print(f"✅ Verified: {len(data)} concepts loaded!")

# Also clean up concept_mesh directory
concept_mesh_dir = Path(r"{PROJECT_ROOT}\concept_mesh")
if concept_mesh_dir.exists():
    for fname in ['concept_mesh_data.json', 'concepts.json']:
        fpath = concept_mesh_dir / fname
        if fpath.exists():
            try:
                with open(fpath, 'r') as f:
                    content = json.load(f)
                    if not content:
                        fpath.unlink()
                        print(f"✅ Removed empty {fname}")
            except:
                pass

print("\n✅ DONE! Now run: python enhanced_launcher.py")
print("You should see: Main concept storage loaded: XX concepts")
