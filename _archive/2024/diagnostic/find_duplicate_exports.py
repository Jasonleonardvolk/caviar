#!/usr/bin/env python3
"""Find duplicate exports in conceptMesh.ts"""

import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def find_duplicate_exports(filepath):
    """Find all export declarations and check for duplicates"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Pattern to match export declarations
    export_pattern = re.compile(r'export\s+(?:const|let|var|function|class|interface|type|enum)\s+(\w+)')
    
    exports = {}
    
    for line_num, line in enumerate(lines, 1):
        match = export_pattern.search(line)
        if match:
            export_name = match.group(1)
            if export_name not in exports:
                exports[export_name] = []
            exports[export_name].append(line_num)
    
    # Find duplicates
    duplicates = {name: lines for name, lines in exports.items() if len(lines) > 1}
    
    if duplicates:
        print("🚨 DUPLICATE EXPORTS FOUND:")
        for name, line_nums in duplicates.items():
            print(f"\n'{name}' exported on lines: {line_nums}")
            for line_num in line_nums:
                print(f"  Line {line_num}: {lines[line_num-1].strip()[:80]}...")
    else:
        print("✅ No duplicate exports found!")
    
    # Specifically check for updateSystemEntropy
    if 'updateSystemEntropy' in exports:
        print(f"\n📍 updateSystemEntropy found on line(s): {exports['updateSystemEntropy']}")

if __name__ == "__main__":
    conceptmesh_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\stores\conceptMesh.ts")
    
    if conceptmesh_path.exists():
        find_duplicate_exports(conceptmesh_path)
    else:
        print(f"❌ File not found: {conceptmesh_path}")
