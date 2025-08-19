#!/usr/bin/env python3
"""
Build concept_mesh_rs wheel with pure Rust backend
No OpenBLAS needed!
"""

import os
import sys
import subprocess
from pathlib import Path

print("🦀 BUILDING CONCEPT MESH WITH PURE RUST")
print("=" * 60)

# Change to concept_mesh directory
concept_mesh_dir = Path("concept_mesh").absolute()
if not concept_mesh_dir.exists():
    print("❌ concept_mesh directory not found!")
    sys.exit(1)

os.chdir(concept_mesh_dir)
print(f"Working in: {os.getcwd()}")

# Step 1: Clean previous builds
print("\n🧹 Cleaning previous builds...")
subprocess.run("cargo clean", shell=True, capture_output=True)
print("✅ Cleaned")

# Step 2: Build the wheel
print("\n📦 Building wheel with maturin...")
print("This may take a few minutes...")

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

result = subprocess.run(
    f'"{sys.executable}" -m maturin build --release',
    shell=True,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',
    env=env
)

if result.returncode != 0:
    print("\n❌ Build failed!")
    print("\nSTDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    sys.exit(1)

print("✅ Build successful!")
print(result.stdout)

# Step 3: Install the wheel
wheel_dir = Path("target/wheels")
wheels = list(wheel_dir.glob("concept_mesh_rs-*.whl"))

if not wheels:
    print("❌ No wheel found!")
    sys.exit(1)

wheel_path = wheels[0]
print(f"\n📦 Installing wheel: {wheel_path.name}")

install_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)],
    capture_output=True,
    text=True
)

print(install_result.stdout)

# Step 4: Test import
print("\n🧪 Testing import...")
os.chdir("..")  # Back to kha directory

test_result = subprocess.run(
    [sys.executable, "-c", """
import concept_mesh_rs
print(f'✅ Imported from: {concept_mesh_rs.__file__}')
print('✅ Pure Rust backend - no OpenBLAS needed!')
"""],
    capture_output=True,
    text=True
)

print(test_result.stdout)
if test_result.returncode != 0:
    print(f"❌ Import failed: {test_result.stderr}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ SUCCESS! Concept Mesh built with pure Rust backend")
print("🚀 No OpenBLAS needed - self-contained wheel installed!")
