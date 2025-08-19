#!/usr/bin/env python3
"""
Test that the runtime path resolution is working correctly
"""
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add scripts to path so we can import iris_paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

print("Testing Runtime Path Resolution")
print("=" * 60)

# Test Python resolution
try:
    from iris_paths import root, resolve, replace_tokens, PROJECT_ROOT
    
    print("\nPython Runtime Tests:")
    print(f"  Project root: {root()}")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  Resolved path: {resolve('config', 'settings.json')}")
    
    test_string = "${IRIS_ROOT}/data/test.txt"
    resolved = replace_tokens(test_string)
    print(f"  Token replacement: {test_string}")
    print(f"                  -> {resolved}")
    
    print("  ✅ Python runtime resolution working!")
    
except Exception as e:
    print(f"  ❌ Python runtime resolution failed: {e}")
    sys.exit(1)

# Test that refactored files work
print("\nChecking Refactored Files:")

# Count files with our markers
markers_found = {
    "{PROJECT_ROOT}": 0,
    "${IRIS_ROOT}": 0,
    "PROJECT_ROOT = Path": 0
}

root_dir = Path(__file__).resolve().parents[2]
checked = 0

for ext in [".py", ".ts", ".js", ".json", ".md"]:
    for file in root_dir.rglob(f"*{ext}"):
        if any(skip in str(file) for skip in ["node_modules", ".venv", ".git", "__pycache__"]):
            continue
        if file.stat().st_size > 2_000_000:
            continue
            
        try:
            content = file.read_text(encoding='utf-8', errors='ignore')
            for marker in markers_found:
                if marker in content:
                    markers_found[marker] += 1
            checked += 1
            
            if checked % 500 == 0:
                print(f"  Checked {checked} files...", end='\r')
                
        except:
            pass

print(f"  Checked {checked} files total        ")
print(f"  Files with {{PROJECT_ROOT}}: {markers_found['{PROJECT_ROOT}']}")
print(f"  Files with ${{IRIS_ROOT}}: {markers_found['${IRIS_ROOT}']}")
print(f"  Files with PROJECT_ROOT header: {markers_found['PROJECT_ROOT = Path']}")

if sum(markers_found.values()) > 0:
    print("  ✅ Refactored markers found in codebase!")
else:
    print("  ⚠️  No refactored markers found - refactoring may not be complete")

# Test environment variable override
print("\nEnvironment Variable Test:")
original_root = str(root())
os.environ['IRIS_ROOT'] = "C:\\TestPath"

# Re-import to test env override
import importlib
import iris_paths
importlib.reload(iris_paths)

new_root = str(iris_paths.root())
if new_root == "C:\\TestPath":
    print(f"  ✅ Environment override working: {new_root}")
else:
    print(f"  ❌ Environment override not working: {new_root}")

# Restore
del os.environ['IRIS_ROOT']

print("\n" + "=" * 60)
print("Runtime path resolution system is active!")
print("\nUsage:")
print("  Python: from scripts.iris_paths import resolve, PROJECT_ROOT")
print("  Node:   import { resolveFS } from '$lib/node/paths'")
print("\nTo check for regressions: tools\\runtime\\CHECK_PATHS.bat")
