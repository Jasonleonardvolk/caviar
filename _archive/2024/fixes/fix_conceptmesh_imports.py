#!/usr/bin/env python3
"""
ConceptMesh Import Fixer - Finds and fixes incorrect imports
"""

import os
import sys
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("🔍 ConceptMesh Import Fixer")
print("=" * 50)

# Project root
project_root = Path(r"{PROJECT_ROOT}")

# Pattern to find incorrect imports
incorrect_patterns = [
    r"from\s+concept_mesh\s+import\s+ConceptMesh",  # from python.core import ConceptMesh
    r"import\s+concept_mesh\.ConceptMesh",          # from python.core import ConceptMesh
]

# Correct import patterns
correct_import = "from python.core import ConceptMesh"

# Files to check
extensions = ['.py']
exclude_dirs = {'__pycache__', '.git', 'node_modules', 'venv', '.venv', 'build', 'dist'}

print(f"📂 Scanning directory: {project_root}")
print(f"🔎 Looking for incorrect imports of ConceptMesh...")
print()

files_with_issues = []

# Scan all Python files
for root, dirs, files in os.walk(project_root):
    # Remove excluded directories
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    
    for file in files:
        if any(file.endswith(ext) for ext in extensions):
            filepath = Path(root) / file
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Check for incorrect patterns
                for pattern in incorrect_patterns:
                    matches = list(re.finditer(pattern, content, re.MULTILINE))
                    if matches:
                        files_with_issues.append({
                            'file': filepath,
                            'pattern': pattern,
                            'matches': matches,
                            'content': content
                        })
                        
                        print(f"❌ Found incorrect import in: {filepath.relative_to(project_root)}")
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            print(f"   Line {line_num}: {match.group()}")
                        print()
                        
            except Exception as e:
                pass  # Skip files we can't read

if not files_with_issues:
    print("✅ No incorrect ConceptMesh imports found!")
else:
    print(f"\n📊 Found {len(files_with_issues)} files with incorrect imports")
    
    # Ask if user wants to fix them
    print("\n🔧 Would you like to automatically fix these imports?")
    print(f"   They will be changed to: {correct_import}")
    response = input("   Fix them? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\n🔧 Fixing imports...")
        
        for issue in files_with_issues:
            filepath = issue['file']
            content = issue['content']
            
            # Replace incorrect imports
            for pattern in incorrect_patterns:
                content = re.sub(pattern, correct_import, content)
            
            # Write back
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Fixed: {filepath.relative_to(project_root)}")
            except Exception as e:
                print(f"❌ Failed to fix {filepath}: {e}")
        
        print("\n✅ Import fixes complete!")
    else:
        print("\n📝 Manual fix instructions:")
        print("   Replace any occurrence of:")
        print("      from python.core import ConceptMesh")
        print("   With:")
        print(f"      {correct_import}")

print("\n🧪 Testing correct import...")
sys.path.insert(0, str(project_root))

try:
    from python.core import ConceptMesh
    print("✅ SUCCESS: ConceptMesh imported correctly!")
    print(f"   Module: {ConceptMesh.__module__}")
    print(f"   Class: {ConceptMesh}")
except ImportError as e:
    print(f"❌ FAILED to import ConceptMesh: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Check that python/core/concept_mesh.py exists")
    print("   2. Check that python/core/__init__.py exports ConceptMesh")
    print("   3. Make sure there are no syntax errors in concept_mesh.py")

print("\n" + "=" * 50)
print("🏁 Diagnostic complete!")
