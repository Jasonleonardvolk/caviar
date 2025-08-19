"""
Set up TORI to use consistent database paths across all components
"""
import os
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the canonical location for concept database
CANONICAL_DB_DIR = Path(r"{PROJECT_ROOT}\ingest_pdf\data")
CANONICAL_DB_DIR.mkdir(parents=True, exist_ok=True)

# Ensure the JSON files exist
concept_db_path = CANONICAL_DB_DIR / "concept_db.json"
universal_seed_path = CANONICAL_DB_DIR / "universal_seed.json"

if not concept_db_path.exists():
    with open(concept_db_path, "w") as f:
        json.dump([], f)
    print(f"✓ Created {concept_db_path}")

if not universal_seed_path.exists():
    with open(universal_seed_path, "w") as f:
        json.dump([], f)
    print(f"✓ Created {universal_seed_path}")

# Create a .env file to set paths for all components
env_content = f"""# TORI Database Configuration
CONCEPT_DB_PATH={concept_db_path}
UNIVERSAL_SEED_PATH={universal_seed_path}
TORI_CONCEPT_DB={CANONICAL_DB_DIR}
"""

with open(".env", "w") as f:
    f.write(env_content)
print("✓ Created .env file with database paths")

# Also set environment variables for current session
os.environ["CONCEPT_DB_PATH"] = str(concept_db_path)
os.environ["UNIVERSAL_SEED_PATH"] = str(universal_seed_path)
os.environ["TORI_CONCEPT_DB"] = str(CANONICAL_DB_DIR)

print(f"""
✅ Database configuration complete!

All components will now use:
- Concept DB: {concept_db_path}
- Universal Seed: {universal_seed_path}

The .env file has been created to persist these settings.
""")
