@echo off
echo Checking concept counts...
echo.

python -c "
import json
from pathlib import Path

def count_concepts(base_path, name):
    print(f'\n{name.upper()} Concept Counts:')
    print('-'*40)
    
    # Check common concept files
    files_to_check = [
        'concepts.json',
        'concept_mesh_data.json',
        'concept_registry.json',
        'concept_registry_enhanced.json',
        'data/concepts.json'
    ]
    
    total = 0
    for file in files_to_check:
        filepath = base_path / file
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                count = 0
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    if 'concepts' in data:
                        count = len(data['concepts'])
                    elif 'items' in data:
                        count = len(data['items'])
                    else:
                        count = len(data)
                
                if count > 0:
                    print(f'  {file}: {count} concepts')
                    total += count
                    
            except Exception as e:
                print(f'  {file}: Error reading ({e})')
    
    if total == 0:
        print('  No concepts found!')
    else:
        print(f'  TOTAL UNIQUE: ~{total} concepts')
    
    return total

# Check both directories
tori = Path(r'C:\Users\jason\Desktop\tori\kha')
pigpen = Path(r'C:\Users\jason\Desktop\pigpen')

tori_count = count_concepts(tori, 'TORI')
pigpen_count = count_concepts(pigpen, 'Pigpen')

print(f'\n🎯 SUMMARY:')
print(f'  TORI has ~{tori_count} concepts')
print(f'  Pigpen has ~{pigpen_count} concepts')

if pigpen_count > tori_count:
    print(f'\n✅ Pigpen has {pigpen_count - tori_count} MORE concepts!')
    print('   Your import fixes in pigpen are working!')
elif pigpen_count == tori_count and pigpen_count > 0:
    print(f'\n✓ Both have the same number of concepts')
else:
    print(f'\n⚠️  TORI has more concepts than pigpen')
"

echo.
pause
