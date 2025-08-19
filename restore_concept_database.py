"""
Restore concept database to proper location with correct filenames
"""
import json
import shutil
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define paths
ROOT_DIR = Path(r"{PROJECT_ROOT}")
TARGET_DIR = ROOT_DIR / "ingest_pdf" / "data"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

print("🔍 Checking for concept mesh data...")

# Check for the main concept mesh data file
concept_mesh_data_path = ROOT_DIR / "concept_mesh_data.json"
if concept_mesh_data_path.exists():
    print(f"✅ Found concept_mesh_data.json with concept data")
    
    # Read the concept mesh data
    with open(concept_mesh_data_path, 'r', encoding='utf-8') as f:
        mesh_data = json.load(f)
    
    # Extract concepts into the format expected by the pipeline
    concepts = []
    if 'concepts' in mesh_data and isinstance(mesh_data['concepts'], list):
        for doc in mesh_data['concepts']:
            if 'concepts' in doc and isinstance(doc['concepts'], list):
                concepts.extend(doc['concepts'])
    
    print(f"📊 Found {len(concepts)} concepts to restore")
    
    # Write to concept_file_storage.json
    target_file = TARGET_DIR / "concept_file_storage.json"
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(concepts, f, indent=2)
    print(f"✅ Wrote {len(concepts)} concepts to {target_file}")
else:
    print("❌ concept_mesh_data.json not found in root directory")

# Check for concepts.json as alternative source
concepts_json_path = ROOT_DIR / "concepts.json"
if concepts_json_path.exists():
    with open(concepts_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data.get('concepts'):
        print(f"📊 Found additional concepts in concepts.json")
        # This file appears to be empty in your case

# Create or copy seed file
seed_target = TARGET_DIR / "concept_seed_universal.json"
if not seed_target.exists():
    # Create a basic seed file with common academic concepts
    seed_concepts = [
        {"name": "quantum mechanics", "score": 0.9, "type": "topic"},
        {"name": "machine learning", "score": 0.9, "type": "topic"},
        {"name": "neural networks", "score": 0.85, "type": "topic"},
        {"name": "artificial intelligence", "score": 0.9, "type": "topic"},
        {"name": "deep learning", "score": 0.85, "type": "topic"},
        {"name": "natural language processing", "score": 0.85, "type": "topic"},
        {"name": "computer vision", "score": 0.85, "type": "topic"},
        {"name": "reinforcement learning", "score": 0.8, "type": "topic"},
    ]
    with open(seed_target, 'w', encoding='utf-8') as f:
        json.dump(seed_concepts, f, indent=2)
    print(f"✅ Created seed file with {len(seed_concepts)} universal concepts")

print("\n✅ Concept database restoration complete!")
print(f"📂 Files are in: {TARGET_DIR}")
print("\nNext steps:")
print("1. Install required packages:")
print("   pip install pydub opencv-python-headless")
print("2. Run: python enhanced_launcher.py")
print("\nThe pipeline should now show:")
print("✅ Main concept file_storage loaded: X concepts")
print("🌍 Universal seed concepts loaded: 8 concepts")
