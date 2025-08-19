from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import re

# Read the file
with open(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\stores\conceptMesh.ts", 'r', encoding='utf-8') as f:
    content = f.read()

# Find all updateSystemEntropy occurrences
matches = list(re.finditer(r'export\s+(?:const|function)\s+updateSystemEntropy', content))

print(f"Found {len(matches)} export(s) of updateSystemEntropy")

for i, match in enumerate(matches):
    # Get line number
    line_num = content[:match.start()].count('\n') + 1
    # Get the line content
    line_start = content.rfind('\n', 0, match.start()) + 1
    line_end = content.find('\n', match.end())
    line_content = content[line_start:line_end if line_end != -1 else None]
    
    print(f"\nOccurrence {i+1}:")
    print(f"  Line {line_num}: {line_content.strip()[:100]}...")
