#!/usr/bin/env python3
"""
Debug why concept_mesh_rs isn't being found
"""

import sys
import subprocess
from pathlib import Path

print("DEBUGGING CONCEPT_MESH_RS IMPORT ISSUE")
print("=" * 60)

# 1. Check Python path
print("\n1. Python executable:")
print(f"   {sys.executable}")

# 2. Check sys.path
print("\n2. Python import paths (first 5):")
for i, p in enumerate(sys.path[:5]):
    print(f"   {i}: {p}")

# 3. Check site-packages
site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
print(f"\n3. Site-packages location:")
print(f"   {site_packages}")
print(f"   Exists: {site_packages.exists()}")

# 4. Look for concept_mesh files
print("\n4. Looking for concept_mesh files:")
if site_packages.exists():
    concept_files = list(site_packages.glob("concept_mesh*"))
    if concept_files:
        for f in concept_files:
            print(f"   Found: {f.name} ({f.stat().st_size} bytes)")
    else:
        print("   No concept_mesh files found!")

# 5. Check wheel location
print("\n5. Checking for built wheel:")
wheel_dir = Path("concept_mesh/target/wheels")
if wheel_dir.exists():
    wheels = list(wheel_dir.glob("*.whl"))
    if wheels:
        for w in wheels:
            print(f"   Found wheel: {w.name} ({w.stat().st_size} bytes)")
    else:
        print("   No wheels found!")
else:
    print("   Wheel directory doesn't exist!")

# 6. Try to import with full error
print("\n6. Attempting import with full error:")
import_test = """
import sys
import traceback
try:
    import concept_mesh_rs
    print(f"SUCCESS: Imported from {concept_mesh_rs.__file__}")
except ImportError as e:
    print("IMPORT ERROR:")
    print(traceback.format_exc())
"""

result = subprocess.run(
    [sys.executable, "-c", import_test],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# 7. Check what's trying to import it
print("\n7. Checking import locations in code:")

# Check enhanced_launcher.py
launcher_path = Path("enhanced_launcher.py")
if launcher_path.exists():
    content = launcher_path.read_text()
    
    # Find import lines
    import_lines = []
    for i, line in enumerate(content.split('\n')):
        if 'concept_mesh' in line.lower() and ('import' in line or 'from' in line):
            import_lines.append((i+1, line.strip()))
    
    if import_lines:
        print("\n   In enhanced_launcher.py:")
        for line_no, line in import_lines[:3]:
            print(f"      Line {line_no}: {line}")

# 8. Force install the wheel
print("\n8. Force installing the wheel:")
wheel_dir = Path("concept_mesh/target/wheels")
if wheel_dir.exists():
    wheels = list(wheel_dir.glob("*.whl"))
    if wheels:
        wheel = wheels[0]
        print(f"   Installing: {wheel.name}")
        
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)],
            capture_output=True,
            text=True
        )
        
        if install_result.returncode == 0:
            print("   [OK] Install completed")
            
            # Test again
            test2 = subprocess.run(
                [sys.executable, "-c", "import concept_mesh_rs; print('WORKS!')"],
                capture_output=True,
                text=True
            )
            
            if test2.returncode == 0:
                print("   [OK] Import now works!")
            else:
                print("   [FAIL] Still can't import:", test2.stderr)
        else:
            print("   [FAIL] Install failed:", install_result.stderr)

print("\n" + "=" * 60)
