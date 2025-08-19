#!/usr/bin/env python3
"""
One-shot build and install concept_mesh_rs wheel
Following the exact steps from the one-pager
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

print("BUILD AND INSTALL CONCEPT_MESH_RS WHEEL")
print("=" * 60)

# 0. Prerequisites check
print("\n0. Checking prerequisites:")
print(f"   Python: {sys.executable}")
print(f"   Working dir: {os.getcwd()}")

# Check maturin
maturin_check = subprocess.run([sys.executable, "-m", "pip", "show", "maturin"], capture_output=True)
if maturin_check.returncode != 0:
    print("   Installing maturin...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "maturin"])
else:
    print("   [OK] Maturin installed")

# 1. Build the wheel
print("\n1. Building the wheel:")
os.chdir("concept_mesh")
print(f"   Working in: {os.getcwd()}")

# Clean previous builds
if Path("target").exists():
    print("   Cleaning previous builds...")
    shutil.rmtree("target", ignore_errors=True)

print("   Running: maturin build --release")
build_result = subprocess.run(
    [sys.executable, "-m", "maturin", "build", "--release"],
    capture_output=False  # Show output
)

if build_result.returncode != 0:
    print("\n[ERROR] Build failed!")
    sys.exit(1)

# 2. Install the wheel
print("\n2. Installing the wheel:")
wheel_pattern = "target/wheels/concept_mesh_rs-*.whl"
wheels = list(Path(".").glob(wheel_pattern))

if not wheels:
    print("[ERROR] No wheel found!")
    sys.exit(1)

wheel_path = wheels[0]
print(f"   Found: {wheel_path}")

install_cmd = [sys.executable, "-m", "pip", "install", str(wheel_path)]
print(f"   Running: pip install {wheel_path.name}")

install_result = subprocess.run(install_cmd, capture_output=True, text=True)
if install_result.returncode == 0:
    print("   [OK] Successfully installed!")
else:
    print(f"[ERROR] Install failed: {install_result.stderr}")
    sys.exit(1)

# Go back to kha directory
os.chdir("..")

# 3. Verify import
print("\n3. Verifying import:")
verify_script = """
import concept_mesh_rs, sys, pathlib
print(f"[OK] Loaded: {pathlib.Path(concept_mesh_rs.__file__).name}")
print(f"Interpreter: {sys.executable}")
"""

verify_result = subprocess.run(
    [sys.executable, "-c", verify_script],
    capture_output=True,
    text=True
)

if verify_result.returncode == 0:
    print(verify_result.stdout)
else:
    print(f"[ERROR] Import failed: {verify_result.stderr}")

# 4. Remove any shadowing .py files
print("\n4. Checking for shadowing files:")
shadow_file = Path("concept_mesh_rs.py")
if shadow_file.exists():
    print(f"   [!] Found shadowing file: {shadow_file}")
    shadow_file.rename("concept_mesh_rs.py.old")
    print("   [OK] Renamed to .old")
else:
    print("   [OK] No shadowing files")

# 5. Final summary
print("\n" + "=" * 60)
print("SUCCESS! concept_mesh_rs wheel built and installed!")
print("\nNext steps:")
print("1. Kill any running Python: taskkill /IM python.exe /F")
print("2. Start the server: python enhanced_launcher.py")
print("\nExpected results:")
print("- Main process: Shows Penrose backend loaded")
print("- MCP subprocess: No mock warnings")
print("- Both use the same .pyd from site-packages")
