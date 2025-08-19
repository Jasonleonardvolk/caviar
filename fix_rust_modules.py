#!/usr/bin/env python3
"""
Fix Rust module conflicts in concept_mesh
"""

import os
from pathlib import Path

print("FIXING RUST MODULE CONFLICTS")
print("=" * 60)

# Change to concept_mesh directory
os.chdir("concept_mesh/src")

# 1. Fix module ambiguity - rename single files
print("\n1. Fixing module ambiguity:")
print("-" * 40)

# Check for conflicting files
conflicts = [
    ("mesh.rs", "mesh/mod.rs", "mesh_legacy.rs"),
    ("psiarc.rs", "psiarc/mod.rs", "psiarc_legacy.rs")
]

for single_file, dir_mod, new_name in conflicts:
    single_path = Path(single_file)
    dir_path = Path(dir_mod)
    
    if single_path.exists() and dir_path.exists():
        print(f"   Found conflict: {single_file} vs {dir_mod}")
        print(f"   Renaming {single_file} -> {new_name}")
        single_path.rename(new_name)
    elif single_path.exists():
        print(f"   Only {single_file} exists (no conflict)")
    elif dir_path.exists():
        print(f"   Only {dir_mod} exists (no conflict)")

# 2. Fix unclosed delimiter in concept_trail.rs
print("\n2. Fixing unclosed delimiter in concept_trail.rs:")
print("-" * 40)

concept_trail = Path("concept_trail.rs")
if concept_trail.exists():
    content = concept_trail.read_text(encoding='utf-8')
    
    # Count braces
    open_braces = content.count('{')
    close_braces = content.count('}')
    
    print(f"   Open braces: {open_braces}")
    print(f"   Close braces: {close_braces}")
    print(f"   Missing: {open_braces - close_braces}")
    
    if open_braces > close_braces:
        # Add missing closing braces at the end
        missing = open_braces - close_braces
        print(f"   Adding {missing} closing brace(s)")
        
        # Backup first
        backup = concept_trail.with_suffix('.rs.backup')
        backup.write_text(content, encoding='utf-8')
        
        # Add closing braces
        content = content.rstrip() + '\n' + ('}\n' * missing)
        concept_trail.write_text(content, encoding='utf-8')
        print("   [OK] Fixed unclosed delimiters")
else:
    print("   [!] concept_trail.rs not found")

# Go back to concept_mesh directory
os.chdir("..")

# 3. Check if it compiles now
print("\n3. Checking if crate compiles:")
print("-" * 40)

import subprocess

check_result = subprocess.run(
    ["cargo", "check", "--release"],
    capture_output=True,
    text=True
)

if check_result.returncode == 0:
    print("   [OK] Cargo check passed!")
else:
    print("   [!] Still has errors:")
    if check_result.stderr:
        # Show first few lines of error
        lines = check_result.stderr.split('\n')[:10]
        for line in lines:
            print(f"      {line}")

# Go back to kha directory
os.chdir("..")

print("\n" + "=" * 60)
print("Fixes applied!")
print("\nNext steps:")
print("1. Run: python build_wheel_onepager.py")
print("2. This will build and install the wheel")
