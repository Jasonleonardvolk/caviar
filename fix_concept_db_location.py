from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""
Fix concept database location for MCP
"""
import shutil
import os
import json

# Create the data directory in MCP's expected location
mcp_data_dir = r"{PROJECT_ROOT}\mcp_metacognitive\data"
os.makedirs(mcp_data_dir, exist_ok=True)

# Check if concept files exist in the main data directory
main_data_dir = r"{PROJECT_ROOT}\ingest_pdf\data"
concept_db_src = os.path.join(main_data_dir, "concept_db.json")
universal_seed_src = os.path.join(main_data_dir, "universal_seed.json")

# Copy files if they exist
if os.path.exists(concept_db_src):
    shutil.copy2(concept_db_src, os.path.join(mcp_data_dir, "concept_db.json"))
    print(f"✓ Copied concept_db.json to MCP directory")
else:
    # Create empty file
    with open(os.path.join(mcp_data_dir, "concept_db.json"), "w") as f:
        json.dump([], f)
    print("✓ Created empty concept_db.json in MCP directory")

if os.path.exists(universal_seed_src):
    shutil.copy2(universal_seed_src, os.path.join(mcp_data_dir, "universal_seed.json"))
    print(f"✓ Copied universal_seed.json to MCP directory")
else:
    # Create empty file
    with open(os.path.join(mcp_data_dir, "universal_seed.json"), "w") as f:
        json.dump([], f)
    print("✓ Created empty universal_seed.json in MCP directory")

print(f"\nConcept database files are now in: {mcp_data_dir}")
print("\nNext steps:")
print("1. Install required packages:")
print("   pip install pydub opencv-python-headless")
print("2. Optionally install recommended packages:")
print('   pip install "gudhi>=3.8,<3.10" "pymupdf>=1.24"')
print("3. Download ffmpeg and add to PATH (for pydub)")
print("4. Run: python enhanced_launcher.py")
