# Fix TORI Blockers Script
Write-Host "`n🔧 Fixing TORI Launch Blockers" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

# 1. Fix NumPy/spaCy compatibility
Write-Host "`n📦 Step 1: Fixing NumPy/spaCy compatibility..." -ForegroundColor Yellow
Write-Host "Downgrading NumPy to 1.26.x for spaCy compatibility" -ForegroundColor Gray

& C:\ALANPY311\Scripts\activate
pip uninstall -y numpy
pip install "numpy<1.27,>=1.26.4"

Write-Host "Reinstalling spaCy model..." -ForegroundColor Gray
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl"

# Verify
Write-Host "`nVerifying NumPy/spaCy..." -ForegroundColor Cyan
python -c "import numpy, spacy; print(f'NumPy {numpy.__version__} | spaCy {spacy.__version__}'); spacy.load('en_core_web_lg'); print('✅ spaCy model loads successfully')"

# 2. Set up concept database files
Write-Host "`n📁 Step 2: Setting up concept database files..." -ForegroundColor Yellow

# Create the directory
$dataDir = "C:\Users\jason\Desktop\tori\kha\ingest_pdf\data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

# Run Python script to create/copy files
python -c @"
import os
import shutil
import json
from pathlib import Path

data_dir = Path(r'C:\Users\jason\Desktop\tori\kha\ingest_pdf\data')

# Check for existing files and copy/rename them
old_files = [
    (data_dir / 'concept_db.json', 'concept_file_storage.json'),
    (data_dir / 'universal_seed.json', 'concept_seed_universal.json'),
]

for old_path, new_name in old_files:
    new_path = data_dir / new_name
    if old_path.exists() and not new_path.exists():
        shutil.copy2(old_path, new_path)
        print(f'✅ Copied {old_path.name} → {new_name}')
    elif new_path.exists():
        print(f'✅ {new_name} already exists')
    else:
        # Create empty file
        with open(new_path, 'w') as f:
            json.dump([], f)
        print(f'✅ Created empty {new_name}')

# Check for concept_mesh directory and clean it up
concept_mesh_dir = Path(r'C:\Users\jason\Desktop\tori\kha\concept_mesh')
if concept_mesh_dir.exists():
    empty_files = ['concept_mesh_data.json', 'concepts.json']
    for filename in empty_files:
        file_path = concept_mesh_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if not data or (isinstance(data, list) and len(data) == 0) or (isinstance(data, dict) and not data):
                        file_path.unlink()
                        print(f'✅ Removed empty {filename} from concept_mesh directory')
                except:
                    pass
"@

Write-Host "`n🎉 Setup complete!" -ForegroundColor Green
Write-Host "`nNext: Run the Python script to fix ingest_image.py syntax error" -ForegroundColor Yellow
Write-Host "Then: python enhanced_launcher.py" -ForegroundColor Yellow
