# Fix TORI Blockers Script
Write-Host "Fixing TORI Launch Blockers" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan

# 1. Fix NumPy/spaCy compatibility
Write-Host "`nStep 1: Fixing NumPy/spaCy compatibility..." -ForegroundColor Yellow

& C:\ALANPY311\Scripts\activate
pip uninstall -y numpy
pip install "numpy<1.27,>=1.26.4"

Write-Host "Reinstalling spaCy model..." -ForegroundColor Gray
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl"

# 2. Run Python script to fix concept files
Write-Host "`nStep 2: Setting up concept database files..." -ForegroundColor Yellow

python -c @"
import os
import shutil
import json
from pathlib import Path

data_dir = Path(r'C:\Users\jason\Desktop\tori\kha\ingest_pdf\data')
data_dir.mkdir(parents=True, exist_ok=True)

# Fix file names
old_to_new = {
    'concept_db.json': 'concept_file_storage.json',
    'universal_seed.json': 'concept_seed_universal.json',
}

for old_name, new_name in old_to_new.items():
    old_path = data_dir / old_name
    new_path = data_dir / new_name
    
    if old_path.exists() and not new_path.exists():
        shutil.copy2(old_path, new_path)
        print(f'Copied {old_name} to {new_name}')
    elif new_path.exists():
        print(f'{new_name} already exists')
    else:
        with open(new_path, 'w') as f:
            json.dump([], f)
        print(f'Created empty {new_name}')

# Clean up concept_mesh directory
concept_mesh = Path(r'C:\Users\jason\Desktop\tori\kha\concept_mesh')
if concept_mesh.exists():
    for fname in ['concept_mesh_data.json', 'concepts.json']:
        fpath = concept_mesh / fname
        if fpath.exists():
            try:
                with open(fpath) as f:
                    data = json.load(f)
                    if not data:
                        fpath.unlink()
                        print(f'Removed empty {fname}')
            except:
                pass
"@

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "The syntax error in ingest_image.py has been fixed." -ForegroundColor Yellow
Write-Host "Now run: python enhanced_launcher.py" -ForegroundColor Cyan
