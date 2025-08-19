#!/usr/bin/env python3
"""
Check why the wheel wasn't built and rebuild it
"""

import os
import sys
import subprocess
from pathlib import Path

print("CHECKING WHEEL BUILD STATUS")
print("=" * 60)

# 1. Check if maturin is installed
print("\n1. Checking maturin:")
maturin_check = subprocess.run(
    [sys.executable, "-m", "pip", "show", "maturin"],
    capture_output=True,
    text=True
)

if maturin_check.returncode == 0:
    print("[OK] Maturin is installed")
else:
    print("[!] Maturin not installed, installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "maturin"])

# 2. Check Cargo.toml
print("\n2. Checking Cargo.toml:")
cargo_path = Path("concept_mesh/Cargo.toml")
if cargo_path.exists():
    print("[OK] Cargo.toml exists")
    
    # Check for problematic dependencies
    content = cargo_path.read_text()
    if "ndarray-linalg" in content and not content.count("# ndarray-linalg"):
        print("[!] ndarray-linalg is still active!")
    else:
        print("[OK] ndarray-linalg is commented out")
else:
    print("[ERROR] Cargo.toml not found!")

# 3. Clean and rebuild
print("\n3. Rebuilding wheel:")
print("-" * 40)

os.chdir("concept_mesh")

# Clean first
print("Cleaning...")
subprocess.run("cargo clean", shell=True, capture_output=True)

# Set environment
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

# Build
print("Building (this will take a few minutes)...")
build_result = subprocess.run(
    f'"{sys.executable}" -m maturin build --release',
    shell=True,
    capture_output=False,  # Show output directly
    env=env
)

if build_result.returncode == 0:
    print("\n[OK] Build completed!")
    
    # Check for wheel
    wheel_dir = Path("target/wheels")
    if wheel_dir.exists():
        wheels = list(wheel_dir.glob("*.whl"))
        if wheels:
            print(f"\n[OK] Found wheel: {wheels[0].name}")
            
            # Install it
            print("\nInstalling wheel...")
            os.chdir("..")
            
            install_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--force-reinstall", str(Path("concept_mesh") / wheels[0].parent.name / wheels[0].name)],
                capture_output=True,
                text=True
            )
            
            if install_result.returncode == 0:
                print("[OK] Installed successfully!")
                
                # Remove the .py stub if it exists
                py_stub = Path("concept_mesh_rs.py")
                if py_stub.exists():
                    print(f"\n[!] Removing Python stub: {py_stub}")
                    py_stub.rename("concept_mesh_rs.py.old")
                
                # Test import
                print("\nTesting import...")
                test = subprocess.run(
                    [sys.executable, "-c", "import concept_mesh_rs; print(f'Imported from: {concept_mesh_rs.__file__}')"],
                    capture_output=True,
                    text=True
                )
                print(test.stdout)
            else:
                print(f"[ERROR] Install failed: {install_result.stderr}")
        else:
            print("[ERROR] No wheel found after build!")
    else:
        print("[ERROR] target/wheels directory not created!")
else:
    print(f"\n[ERROR] Build failed with code {build_result.returncode}")
    print("Check the output above for errors")

os.chdir("..")

print("\n" + "=" * 60)
